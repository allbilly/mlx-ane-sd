"""Separate native ANE wait, mapped-buffer transfers and CPU draft work.

This is a diagnostic with clock reads and ctypes snapshots, not a replacement
throughput benchmark. Four real benchmark prefixes prime the trained draft.
The same proposal is repeated without committing new tokens.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from asahi_ane import ANE, REPO
from asahi_dflash import DFlash, rope
from bench_asahi_dflash import device_locks, target_forward
from bench_asahi_mlx import TARGET, TARGET_REV, model_snapshot, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "notes/asahi/dflash_native_profile.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--fixture-out", type=Path, default=REPO / ".asahi/macos-capture-inputs")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    os.environ["ANE_PROFILE"] = "1"
    path, meta = model_snapshot(TARGET, TARGET_REV)
    bench_path = REPO / "notes/asahi/dflash_bf16_serial.json"
    bench = json.loads(bench_path.read_text())
    if bench["target"] != meta:
        raise ValueError("Benchmark target differs")
    report = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "procedure": __doc__, "target": meta, "mlx_provenance": bench["mlx_provenance"],
              "library_sha256": sha256(REPO / ".asahi/build/libane_sd.so"),
              "sources": {p.name: sha256(p) for p in (Path(__file__), REPO / "scripts/asahi_ane.py",
                  REPO / "scripts/asahi_dflash.py", REPO / "asahi/vendor/qwen3.c/ane/ane_matmul.c")},
              "rows": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.fixture_out.mkdir(parents=True, exist_ok=True)
    with device_locks(), ANE() as ane:
        draft = DFlash(REPO / ".asahi/models/microcycle-dflash", ane)
        model, tok = load(str(path))
        captured, totals = {}, {}
        def wrap(label, fn):
            def call(x):
                before = ane.profile
                out = fn(x)
                after = ane.profile
                delta = {k: after[k] - before[k] for k in before}
                previous = totals.setdefault(label, {k: 0 for k in delta})
                for k, value in delta.items():
                    previous[k] += value
                if label in ("layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
                             "layers.0.self_attn.v_proj.weight"):
                    captured[label] = out.copy()
                if label == "head.0":
                    captured["head_input"] = x.copy()
                return out
            return call
        draft.linears = {name: wrap(name, fn) for name, fn in draft.linears.items()}
        draft.head_chunks = [(base, wrap(f"head.{base}", fn)) for base, fn in draft.head_chunks]
        original_norm = draft.norm
        def norm(x, name):
            if name in ("layers.0.input_layernorm", "layers.0.post_attention_layernorm"):
                captured[name] = x.copy()
            return original_norm(x, name)
        draft.norm = norm
        for source in bench["rows"][:4]:
            draft.reset()
            cache = make_prompt_cache(model)
            teacher = source["baseline"]["tokens"]
            history = tok.encode(source["prompt"]) + teacher[:16]
            for begin in range(0, len(history), 32):
                _, features = target_forward(model, history[begin:begin + 32], cache, draft.feature_ids, logits=False)
                draft.append(features)
            anchor = teacher[16]
            draft.propose(anchor)
            for repeat in range(args.repeats):
                totals.clear()
                ane.reset_profile()
                draft.profile = dict.fromkeys(draft.profile, 0.0)
                start = time.perf_counter()
                proposals, _ = draft.propose(anchor)
                wall_ms = (time.perf_counter() - start) * 1000
                row = {"name": source["name"], "repeat": repeat, "context_tokens": draft.offset,
                       "wall_ms": wall_ms, "draft_phase_s": dict(draft.profile),
                       "native": ane.profile, "kernels": {k: dict(v) for k, v in totals.items()},
                       "proposals": proposals}
                report["rows"].append(row)
                print(f"{source['name']} {repeat}: wall={wall_ms:.2f} ms "
                      f"ioctl={row['native']['submit_ns']/1e6:.2f} ms calls={row['native']['calls']}", flush=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
            positions = np.arange(draft.offset, draft.offset + draft.block_size)
            attn = "layers.0.self_attn."
            q = rope(original_norm(captured[attn + "q_proj.weight"].reshape(-1, draft.heads, draft.head_dim),
                                   attn + "q_norm"), positions, draft.theta)
            k = rope(original_norm(captured[attn + "k_proj.weight"].reshape(-1, draft.kv_heads, draft.head_dim),
                                   attn + "k_norm"), positions, draft.theta)
            v = captured[attn + "v_proj.weight"].reshape(-1, draft.kv_heads, draft.head_dim)
            context_k, context_v, _ = draft.cache[0]
            k, v = np.concatenate((context_k, k)), np.concatenate((context_v, v))
            repeat_heads = draft.heads // draft.kv_heads
            fixture = args.fixture_out / (source["name"] + ".npz")
            np.savez(fixture, norm_input=captured["layers.0.input_layernorm"],
                     ffn_input=captured["layers.0.post_attention_layernorm"],
                     head_input=captured["head_input"], q=q.transpose(1, 2, 0),
                     k=np.repeat(k, repeat_heads, axis=1).transpose(1, 2, 0),
                     v=np.repeat(v, repeat_heads, axis=1).transpose(1, 2, 0),
                     context_tokens=np.array(draft.offset), anchor=np.array(anchor))
            report.setdefault("fixtures", {})[source["name"]] = {"file": fixture.name,
                                                                   "sha256": sha256(fixture)}
            del cache
        report["fixture_source"] = {"draft": json.loads((draft.path / "SOURCE.json").read_text()),
                                    "feature_ids": draft.feature_ids,
                                    "weights_sha256": sha256(draft.path / "model.safetensors")}
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        (args.fixture_out / "SOURCE.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
