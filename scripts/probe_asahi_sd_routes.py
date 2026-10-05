"""Measure real-prefix target calls and real-input linear routes on Vulkan.

Uses the four 100-token scalar baseline continuations already recorded in
dflash_bf16_serial.json. All target calls start with the same teacher-forced
cache history. Times include synchronous evaluation, feature extraction, and
cache evaluation as in the DFlash runner. Padding and row-GEMV are diagnostics.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

from asahi_ane import REPO
from asahi_verify_routes import linear_route, matmul
from bench_asahi_dflash import device_locks, target_forward
from bench_asahi_mlx import TARGET, TARGET_REV, memory_state, model_snapshot, sha256


def timed(fn, repeats):
    mx.eval(fn())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        mx.eval(value)
        samples.append((time.perf_counter() - start) * 1000)
    return value, {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", type=Path, default=REPO / "notes/asahi/dflash_bf16_serial.json")
    ap.add_argument("--out", type=Path, default=REPO / "notes/asahi/sd_routes.json")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    bench = json.loads(args.bench.read_text())
    path, meta = model_snapshot(TARGET, TARGET_REV)
    if meta != bench["target"]:
        raise ValueError("Source benchmark target differs")
    sys.path.insert(0, str(REPO / "asahi/vendor/omarchy-mlx/scripts"))
    from mlx_provenance import installed_provenance
    provenance = installed_provenance()
    if provenance["verified"] != "match" or provenance != bench["mlx_provenance"]:
        raise ValueError("Installed MLX differs from source benchmark")
    result = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "procedure": __doc__, "target": meta, "mlx_provenance": bench["mlx_provenance"],
              "source_bench_sha256": sha256(args.bench),
              "sources": {p.name: sha256(p) for p in (Path(__file__), REPO / "scripts/asahi_verify_routes.py")},
              "memory_before": memory_state(), "target_rows": [], "linear_rows": []}
    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    save()
    with device_locks():
        model, tok = load(str(path))
        for row in bench["rows"][:4]:
            name = row["name"]
            teacher = row["baseline"]["tokens"]
            history = tok.encode(row["prompt"]) + teacher[:16]
            continuation = teacher[16:48]
            def primed_cache():
                cache = make_prompt_cache(model)
                for begin in range(0, len(history), 32):
                    target_forward(model, history[begin:begin + 32], cache, logits=False)
                return cache
            cache = primed_cache()
            reference = []
            for token in continuation:
                ids, _ = target_forward(model, [token], cache, (1, 13, 25))
                reference.extend(ids)
            del cache
            for width in (1, 2, 4, 8, 16, 32):
                modes = ("stock",) if width in (1, 32) else ("stock", "row-gemv", "pad32")
                for mode in modes:
                    cache = primed_cache()
                    predictions, samples = None, []
                    with linear_route(mode):
                        for repeat in range(args.repeats + 1):
                            start = time.perf_counter()
                            predictions, _ = target_forward(model, continuation[:width], cache, (1, 13, 25))
                            elapsed = (time.perf_counter() - start) * 1000
                            trim_prompt_cache(cache, width)
                            if repeat:
                                samples.append(elapsed)
                    item = {"name": name, "width": width, "route": mode,
                            "median_ms": statistics.median(samples), "samples_ms": samples,
                            "tokens_equal": predictions == reference[:width],
                            "scalar_predictions": reference[:width], "block_predictions": predictions}
                    result["target_rows"].append(item)
                    print(f"target {name} M={width} {mode}: {item['median_ms']:.2f} ms identity={item['tokens_equal']}", flush=True)
                    del cache
                    save()
            # Exact real model activations, rather than random input vectors.
            first = model.model.layers[0]
            cache = primed_cache()
            h = model.model.embed_tokens(mx.array(continuation[:8], mx.uint32)[None])
            from mlx_lm.models.base import create_attention_mask
            from mlx_lm.models.activations import swiglu
            qin = first.input_layernorm(h)
            residual = h + first.self_attn(qin, create_attention_mask(h, cache[0]), cache[0])
            gatein = first.post_attention_layernorm(residual)
            downin = swiglu(first.mlp.gate_proj(gatein), first.mlp.up_proj(gatein))
            h = residual + first.mlp.down_proj(downin)
            for layer, c in zip(model.model.layers[1:], cache[1:]):
                h = layer(h, create_attention_mask(h, c), c)
            headin = model.model.norm(h)
            cases = (("q_proj", qin, first.self_attn.q_proj.weight),
                     ("gate_proj", gatein, first.mlp.gate_proj.weight),
                     ("down_proj", downin, first.mlp.down_proj.weight),
                     ("lm_head", headin, model.model.embed_tokens.weight))
            mx.eval([c.state for c in cache], [x for _, x, _ in cases])
            for label, x, weight in cases:
                sequential = mx.concatenate([x[:, i:i + 1] @ weight.T for i in range(8)], axis=1)
                mx.eval(sequential)
                for mode in ("stock", "row-gemv", "pad32"):
                    out, timing = timed(lambda: matmul(x, weight, mode), args.repeats)
                    bits_equal = bool(mx.all(out.view(mx.uint16) == sequential.view(mx.uint16)).item())
                    delta = (out.astype(mx.float32) - sequential.astype(mx.float32))
                    item = {"name": name, "projection": label, "input_shape": list(x.shape),
                            "weight_shape": list(weight.shape), "route": mode, **timing,
                            "bit_equal_to_scalar": bits_equal,
                            "max_abs_difference": float(mx.max(mx.abs(delta)).item())}
                    result["linear_rows"].append(item)
                    print(f"linear {name} {label} {mode}: {item['median_ms']:.3f} ms bits={bits_equal}", flush=True)
                save()
            del cache, cases, h, qin, residual, gatein, downin, headin
        result["memory_after"] = memory_state()
        save()


if __name__ == "__main__":
    main()
