"""Paired native ANE DFlash + MLX/Metal target benchmark on macOS."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import bench_macos_dflash as bench
from contextlib import contextmanager


@contextmanager
def matched_wired_limit(mx, result):
    """Use the same residency budget stock mlx-lm uses during stream_generate."""
    recommended = mx.device_info()["max_recommended_working_set_size"]
    previous = mx.set_wired_limit(recommended)
    result["mlx_wired_limit_bytes"] = {"previous": previous, "during_bench": recommended,
                                       "policy": "same as stock mlx-lm wired_limit"}
    try:
        yield
    finally:
        mx.synchronize()
        mx.set_wired_limit(previous)


def validate_draft(model, tok, draft, path):
    """Gate timing on nonzero real-input agreement, including the vocabulary head."""
    import numpy as np
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from dflash_mlx_speculators import DFlash
    reference = DFlash(path, precision="fp16")
    rows = []
    for label, prompt in bench.PROMPTS:
        ids, features = bench.target_forward(model, tok.encode(prompt), make_prompt_cache(model), draft.feature_ids)
        draft.reset(); reference.reset()
        draft.append(features); reference.append(features)
        actual_ids, actual = draft.propose(ids[-1])
        expected_ids, expected = reference.propose(ids[-1])
        actual = np.asarray(actual, np.float32)
        expected = np.array(expected).astype(np.float32)
        cosine = float(np.sum(actual * expected) / max(1e-30, float(np.linalg.norm(actual) * np.linalg.norm(expected))))
        rows.append({"name": label, "native_ids": actual_ids, "mlx_fp16_ids": expected_ids,
                     "cosine": cosine, "rmse": float(np.sqrt(np.mean((actual - expected)**2))),
                     "first_candidate_matches": actual_ids[0] == expected_ids[0]})
    del reference
    mx.clear_cache()
    return {"rows": rows, "qualified": all(r["cosine"] > .995 for r in rows)
            and sum(r["first_candidate_matches"] for r in rows) >= 3}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, default=Path.home() / "Desktop/ANEForge")
    ap.add_argument("--target", type=Path, default=bench.REPO / ".asahi/m1-sd-macos/target")
    ap.add_argument("--draft", type=Path, default=bench.REPO / ".asahi/models/microcycle-dflash")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--out", type=Path, default=bench.REPO / "notes/m1_macos_ane_dflash.json")
    args = ap.parse_args()
    mx = bench.require_metal()
    if args.max_new < 2 or args.repeats < 1:
        ap.error("Need max-new>=2, repeats>=1")
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "waiting_for_devices",
              "target": {"model": "mlx-community/Qwen3-0.6B-bf16", "revision": "42096995f6402fde107068cf530136fe64b604f8"},
              "draft": json.loads((args.draft / "SOURCE.json").read_text()), "machine": mx.device_info(),
              "max_new": args.max_new, "repeats": args.repeats, "capacity": args.capacity,
              "placement": {"draft_body": "ANE-only e5rt mask 0x4", "target": "MLX Metal bf16"},
              "source_sha256": {name: bench.sha256(bench.REPO / "scripts" / name) for name in
                                ("bench_macos_ane_dflash.py", "dflash_macos_ane.py", "bench_macos_dflash.py",
                                 "dflash_mlx_speculators.py")},
              "configs": []}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    save()
    with bench.device_locks(), matched_wired_limit(mx, result):
        from mlx_lm import load
        from dflash_macos_ane import DFlashANE
        result["status"] = "running"
        result["host_before"] = bench.host_state()
        result["target"]["weights_sha256"] = bench.sha256(args.target / "model.safetensors")
        result["draft"]["weights_sha256"] = bench.sha256(args.draft / "model.safetensors")
        if result["target"]["weights_sha256"] != "bfe05e58f5acdecd38e5f3e64c82071682683a73db0e57fd437523885a49f2ff":
            raise RuntimeError("Pinned bf16 target changed")
        model, tok = load(str(args.target)); model.eval()
        if model.model.embed_tokens.weight.dtype != mx.bfloat16:
            raise RuntimeError("Expected unchanged bf16 target")
        save()
        for head, verification in (("gpu", "batch"), ("ane", "batch"), ("ane", "serial")):
            name = f"body_ane_head_{head}_{verification}"
            config = {"name": name, "head": head, "verification": verification, "rows": []}
            result["configs"].append(config); save()
            print(f"[configuration] {name}", flush=True)
            draft = None
            try:
                begin = time.perf_counter()
                draft = DFlashANE(args.draft, args.reference, args.capacity, head)
                config["load_compile_s"] = time.perf_counter() - begin
                config["runtime"] = draft.provenance
                config["ane_device_mask"] = draft.body._prog._device_mask
                config["fused_body_ops"] = draft.body.n_ops
                print(f"[placement] ANE mask=0x4, body ops={draft.body.n_ops}, head={head}", flush=True)
                config["draft_parity"] = validate_draft(model, tok, draft, args.draft)
                save()
                if not config["draft_parity"]["qualified"]:
                    raise RuntimeError("Real-input ANE draft parity failed; refusing to benchmark")
                print("[parity] real-input ANE draft passed on four prompts", flush=True)
                for spec in (False, True):
                    bench.generate(model, tok, draft, "The weather is", 12, spec, verification)
                bench.stock_baseline(model, tok, "The weather is", 12)
                for repeat in range(args.repeats):
                    for label, prompt in bench.PROMPTS:
                        row = {"repeat": repeat, "name": label, "prompt": prompt}
                        for spec in ((False, True) if repeat % 2 == 0 else (True, False)):
                            mode = "dflash" if spec else "baseline"
                            row[mode] = bench.generate(model, tok, draft, prompt, args.max_new, spec, verification)
                            if spec:
                                row[mode]["ane_evaluations"] = draft.ane_evaluations
                                if not draft.ane_evaluations:
                                    raise RuntimeError("No ANE evaluations: refusing to report offload")
                            print(f"{repeat} {label} {mode}: {row[mode]['tok_per_s_decode']:.2f} tok/s", flush=True)
                        row["stock_mlx_lm"] = bench.stock_baseline(model, tok, prompt, args.max_new)
                        row["token_identity"] = row["dflash"]["tokens"] == row["baseline"]["tokens"]
                        row["stock_token_identity"] = row["baseline"]["tokens"] == row["stock_mlx_lm"]["tokens"]
                        row["speedup_vs_stock"] = row["dflash"]["tok_per_s_decode"] / row["stock_mlx_lm"]["generation_tps"]
                        config["rows"].append(row); save()
                        print(f"  identity={row['token_identity']} ANE calls={row['dflash']['ane_evaluations']} "
                              f"stock_speedup={row['speedup_vs_stock']:.3f}x", flush=True)
                rows = config["rows"]
                config["summary"] = {"stock_mean_tps": statistics.mean(r["stock_mlx_lm"]["generation_tps"] for r in rows),
                    "dflash_mean_tps": statistics.mean(r["dflash"]["tok_per_s_decode"] for r in rows),
                    "mean_speedup_vs_stock": statistics.mean(r["speedup_vs_stock"] for r in rows),
                    "max_speedup_vs_stock": max(r["speedup_vs_stock"] for r in rows),
                    "identity_trials": sum(r["token_identity"] for r in rows), "trials": len(rows),
                    "qualified": config["draft_parity"]["qualified"] and all(
                        r["token_identity"] and r["stock_token_identity"] for r in rows),
                    "ane_evaluations": sum(r["dflash"]["ane_evaluations"] for r in rows)}
                config["status"] = "complete"
                print(json.dumps(config["summary"], indent=2), flush=True)
            except Exception as error:
                config.update(status="failed", error=repr(error)); save()
                print(f"[failed] {name}: {error}", flush=True)
            finally:
                if draft is not None:
                    draft.close()
                save()
        result["host_after"] = bench.host_state()
    result["status"] = "complete" if all(c["status"] == "complete" for c in result["configs"]) else "incomplete"
    result["finished_utc"] = datetime.now(timezone.utc).isoformat(); save()
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
