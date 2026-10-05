"""Real trained DFlash drafting on Linux ANE with a dense MLX/Vulkan target.

Compare four raw prompts using fresh caches, greedy sampling, one full warmup,
and alternating baseline/SD order. Decode timing includes the last prompt-token
forward, draft context updates, all verification, transfers and cache trimming.
Exit 2 means generation completed but diverged from scalar greedy target output.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from asahi_ane import ANE, REPO
from asahi_dflash import DFlash
from bench_final_stack import PROMPTS
from bench_asahi_mlx import TARGET, TARGET_REV, model_snapshot, memory_state, power_state, sha256


@contextlib.contextmanager
def device_locks():
    # Same lock order as the native qwen3.c benchmark queue.
    with contextlib.ExitStack() as stack:
        for name in ("ane.lock", "gpu.lock"):
            lock = stack.enter_context((Path.home() / name).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[wait] another workload holds {name}", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def target_forward(model, ids, cache, feature_ids=(), logits=True):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    h = model.model.embed_tokens(mx.array(ids, mx.uint32)[None])
    mask = create_attention_mask(h, cache[0])
    features = {}
    for i, (layer, c) in enumerate(zip(model.model.layers, cache), 1):
        h = layer(h, mask, c)
        if i in feature_ids:
            features[i] = h
    feature = mx.concatenate([features[i] for i in feature_ids], axis=-1) if feature_ids else None
    if logits:
        norm = model.model.norm(h)
        scores = model.model.embed_tokens.as_linear(norm) if model.args.tie_word_embeddings else model.lm_head(norm)
        tokens = mx.argmax(scores, axis=-1)[0]
    else:
        tokens = None
    mx.eval([c.state for c in cache], *([tokens] if tokens is not None else []),
            *([feature] if feature is not None else []))
    return (tokens.tolist() if tokens is not None else None,
            np.array(feature[0].astype(mx.float32)) if feature is not None else None)


def generate(model, tokenizer, draft, prompt, limit, speculative, verification="batch"):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Prompt must contain at least one token")
    cache = make_prompt_cache(model)
    captures = draft.feature_ids if speculative else ()
    if speculative:
        draft.reset()
    mx.synchronize()
    start = time.perf_counter()
    # Prime in bounded chunks; the final prompt token starts decode timing.
    for begin in range(0, len(ids) - 1, 32):
        _, features = target_forward(model, ids[begin:min(begin + 32, len(ids) - 1)],
                                     cache, captures, logits=False)
        if speculative:
            draft.append(features)
    mx.synchronize()
    prefill_s = time.perf_counter() - start
    if speculative:
        draft.profile = {k: 0.0 for k in draft.profile}
    start = time.perf_counter()
    first, features = target_forward(model, ids[-1:], cache, captures)
    tokens = [first[0]]
    if speculative:
        draft.append(features)
    accepted = proposed = cycles = 0
    verify_s = 0.0
    traces = []
    while len(tokens) < limit and tokens[-1] not in tokenizer.eos_token_ids:
        if not speculative:
            prediction, _ = target_forward(model, tokens[-1:], cache)
            tokens.append(prediction[0])
            continue
        position = cache[0].offset
        if position != draft.offset:
            raise RuntimeError("Target and committed draft cache positions disagree")
        candidates = draft.propose(tokens[-1])[0] if limit - len(tokens) > 1 else []
        # Avoid verifying candidate positions that cannot be returned.
        candidates = candidates[:max(0, limit - len(tokens) - 1)]
        begin = time.perf_counter()
        if verification == "serial":
            predictions, feature_rows = [], []
            for i, token in enumerate([tokens[-1], *candidates]):
                prediction, feature = target_forward(model, [token], cache, captures)
                predictions.extend(prediction)
                feature_rows.append(feature)
                if i < len(candidates) and prediction[0] != candidates[i]:
                    break
            features = np.concatenate(feature_rows)
        else:
            predictions, features = target_forward(model, [tokens[-1], *candidates], cache, captures)
        verify_s += time.perf_counter() - begin
        count = 0
        for candidate, target in zip(candidates, predictions):
            if candidate != target:
                break
            count += 1
        committed = [*candidates[:count], predictions[count]]
        # Stop at EOS within an accepted block; all later positions are rejected.
        for i, token in enumerate(committed):
            if token in tokenizer.eos_token_ids:
                committed = committed[:i + 1]
                break
        # The anchor and accepted proposals have target K/V; the correction is
        # the next unprocessed anchor. It must not enter either cache yet.
        keep = len(committed)
        trim_prompt_cache(cache, len(predictions) - keep)
        draft.append(features[:keep])
        if any(c.offset != position + keep for c in cache) or draft.offset != position + keep:
            raise RuntimeError("Rejected target cache positions were not rolled back")
        tokens.extend(committed)
        traces.append({"position": position, "proposals": candidates,
                       "target_predictions": predictions, "accepted_prefix": count,
                       "committed": committed})
        accepted += min(count, len(committed))
        proposed += len(candidates)
        cycles += 1
    mx.synchronize()
    elapsed = time.perf_counter() - start
    return {"tokens": tokens, "text": tokenizer.decode(tokens), "prompt_tokens": len(ids),
            "prefill_s": prefill_s, "decode_s": elapsed, "tok_per_s_decode": len(tokens) / elapsed,
            "cycles": cycles, "accepted": accepted, "proposed": proposed,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "tokens_per_cycle": (len(tokens) - 1) / cycles if cycles else None,
            "target_verify_s": verify_s,
            "verification_trace": traces,
            "draft_profile_s": dict(draft.profile) if speculative else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, default=REPO / ".asahi/models/microcycle-dflash")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--target-precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--verify", choices=("batch", "serial"), default="batch",
                    help="batch tests SD acceleration; serial preserves scalar bf16 decisions")
    ap.add_argument("--out", type=Path, default=REPO / "notes/asahi/dflash.json")
    args = ap.parse_args()
    if args.max_new < 2 or args.repeats < 1:
        ap.error("--max-new must be >=2 and --repeats >=1")
    import mlx.core as mx
    from mlx_lm import load
    if mx.default_device() != mx.Device(mx.gpu):
        raise RuntimeError("Expected real MLX Vulkan GPU execution")
    if any(os.environ.get(k) for k in ("MLX_OMARCHY_ALLOW_NON_APPLE", "MLX_OMARCHY_CAPS_SIM")):
        raise RuntimeError("Simulated hardware capabilities are not allowed")
    target_path, target_meta = model_snapshot(TARGET, TARGET_REV)
    draft_config = json.loads((args.draft / "config.json").read_text())
    if draft_config["speculators_config"]["verifier"]["name_or_path"] != "Qwen/Qwen3-0.6B":
        raise RuntimeError("Draft was not trained for the pinned Qwen3-0.6B target")
    import sys
    sys.path.insert(0, str(REPO / "asahi/vendor/omarchy-mlx/scripts"))
    from mlx_provenance import installed_provenance
    provenance = installed_provenance()
    if provenance["verified"] != "match":
        raise RuntimeError("Installed MLX fork provenance mismatch")
    result = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "machine": Path("/proc/device-tree/model").read_text().rstrip("\0"),
              "kernel": platform.release(), "mlx_provenance": provenance,
              "target": target_meta, "target_precision": args.target_precision,
              "draft": {"source": json.loads((args.draft / "SOURCE.json").read_text()),
                        "config": draft_config, "weights_sha256": sha256(args.draft / "model.safetensors")},
              "native_library_sha256": sha256(REPO / ".asahi/build/libane_sd.so"),
              "sources": {p.name: sha256(p) for p in (Path(__file__),
                          REPO / "scripts/asahi_ane.py", REPO / "scripts/asahi_dflash.py",
                          REPO / "asahi/vendor/qwen3.c/ane/ane_matmul.c")},
              "max_new": args.max_new, "repeats": args.repeats, "verification": args.verify,
              "procedure": __doc__, "memory_before": memory_state(), "power_before": power_state(), "rows": []}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    save()
    print("[load] trained 0.6B DFlash on ANE; target on Vulkan", flush=True)
    with device_locks(), ANE() as ane:
        draft = DFlash(args.draft, ane)
        model, tok = load(str(target_path))
        if args.target_precision == "fp32":
            model.set_dtype(mx.float32)
            mx.eval(model.parameters())
        print("[warmup] both paths", flush=True)
        for speculative in (False, True):
            generate(model, tok, draft, "The weather is", 8, speculative, args.verify)
        result["warmup_submissions"] = ane.submissions
        for repeat in range(args.repeats):
            for name, prompt in PROMPTS:
                paths = (False, True) if repeat % 2 == 0 else (True, False)
                row = {"repeat": repeat, "name": name, "prompt": prompt}
                before = ane.submissions
                for speculative in paths:
                    mode = "dflash" if speculative else "baseline"
                    row[mode] = generate(model, tok, draft, prompt, args.max_new, speculative, args.verify)
                    print(f"{repeat} {name} {mode}: {row[mode]['tok_per_s_decode']:.2f} tok/s", flush=True)
                row["ane_submissions"] = ane.submissions - before
                row["token_identity"] = row["baseline"]["tokens"] == row["dflash"]["tokens"]
                row["speedup"] = row["dflash"]["tok_per_s_decode"] / row["baseline"]["tok_per_s_decode"]
                row["first_difference"] = next((i for i, pair in enumerate(zip(
                    row["baseline"]["tokens"], row["dflash"]["tokens"])) if pair[0] != pair[1]), None)
                result["rows"].append(row)
                save()
                print(f"  identity={row['token_identity']} speedup={row['speedup']:.3f}x "
                      f"tokens/cycle={row['dflash']['tokens_per_cycle']}", flush=True)
        rows = result["rows"]
        result["summary"] = {"baseline_mean_tps": statistics.mean(r["baseline"]["tok_per_s_decode"] for r in rows),
                             "dflash_mean_tps": statistics.mean(r["dflash"]["tok_per_s_decode"] for r in rows),
                             "mean_speedup": statistics.mean(r["speedup"] for r in rows),
                             "max_speedup": max(r["speedup"] for r in rows),
                             "all_token_identity": all(r["token_identity"] for r in rows),
                             "ane_submissions": ane.submissions}
        result["memory_after"] = memory_state()
        result["power_after"] = power_state()
        save()
    print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["summary"]["all_token_identity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
