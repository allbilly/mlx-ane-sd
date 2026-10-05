"""Matched Qwen3-0.6B target measurements on the same M1 under macOS.

The scalar loop and feature-capturing target calls follow the Linux runner's
synchronous evaluation contract. Teacher-forced verification probes isolate
query width without a draft or speculative acceptance loop.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        ap.error("Requires Apple Silicon macOS")
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    root = Path(__file__).resolve().parent.parent
    manifest = json.loads((root / "manifest.json").read_text())
    model_path = args.model or root / "target"
    out_path = args.out or root / "metal-reference.json"
    if out_path.exists():
        ap.error("--out must not already exist")
    for file, expected in manifest["target"]["files"].items():
        with (model_path / file).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != expected:
                raise ValueError(f"Target differs: {file}")
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    if not mx.metal.is_available() or mx.default_device() != mx.Device(mx.gpu):
        raise RuntimeError("Expected real Metal GPU")
    model, tok = load(str(model_path))
    def forward(ids, cache, captures=False, logits=True):
        h = model.model.embed_tokens(mx.array(ids, mx.uint32)[None])
        mask = create_attention_mask(h, cache[0])
        features = []
        for i, (layer, c) in enumerate(zip(model.model.layers, cache), 1):
            h = layer(h, mask, c)
            if captures and i in (1, 13, 25):
                features.append(h)
        features = mx.concatenate(features, axis=-1) if captures else None
        if logits:
            norm = model.model.norm(h)
            tokens = mx.argmax(model.model.embed_tokens.as_linear(norm), axis=-1)[0]
        else:
            tokens = None
        mx.eval([c.state for c in cache], *([tokens] if tokens is not None else []),
                *([features] if features is not None else []))
        if features is not None:
            # Include the same GPU-to-host feature transfer as Linux SD.
            np.array(features[0].astype(mx.float32))
        return tokens.tolist() if tokens is not None else None
    def prime(history):
        cache = make_prompt_cache(model)
        for begin in range(0, len(history), 32):
            forward(history[begin:begin + 32], cache, logits=False)
        return cache
    def generate(prompt, limit):
        ids = tok.encode(prompt)
        cache = prime(ids[:-1])
        mx.synchronize()
        start = time.perf_counter()
        tokens = forward(ids[-1:], cache)
        while len(tokens) < limit and tokens[-1] not in tok.eos_token_ids:
            tokens.extend(forward(tokens[-1:], cache))
        mx.synchronize()
        elapsed = time.perf_counter() - start
        return {"tokens": tokens, "decode_s": elapsed, "tok_per_s_decode": len(tokens) / elapsed}
    generate("The weather is", 8)
    report = {"schema": 1, "procedure": __doc__, "platform": platform.platform(),
              "target": manifest["target"], "packages": {p: importlib.metadata.version(p)
                  for p in ("mlx", "mlx-lm", "numpy")}, "baseline_rows": [], "verify_rows": []}
    def save():
        out_path.write_text(json.dumps(report, indent=2) + "\n")
    for trace in manifest["teacher_traces"]:
        for repeat in range(args.repeats):
            measured = generate(trace["prompt"], 100)
            measured.update(name=trace["name"], repeat=repeat,
                            token_identity_to_linux=measured["tokens"] == trace["tokens"])
            report["baseline_rows"].append(measured)
            print(f"{trace['name']} {repeat}: {measured['tok_per_s_decode']:.2f} tok/s "
                  f"Linux identity={measured['token_identity_to_linux']}", flush=True)
            save()
        history = tok.encode(trace["prompt"]) + trace["tokens"][:16]
        inputs = trace["tokens"][16:48]
        cache = prime(history)
        scalar = []
        for token in inputs:
            scalar.extend(forward([token], cache, captures=True))
        for width in (1, 2, 4, 8, 16, 32):
            cache = prime(history)
            samples = []
            for repeat in range(4):
                start = time.perf_counter()
                prediction = forward(inputs[:width], cache, captures=True)
                elapsed = (time.perf_counter() - start) * 1000
                trim_prompt_cache(cache, width)
                if repeat:
                    samples.append(elapsed)
            report["verify_rows"].append({"name": trace["name"], "width": width,
                "median_ms": statistics.median(samples), "samples_ms": samples,
                "token_identity_to_scalar": prediction == scalar[:width],
                "block_predictions": prediction, "scalar_predictions": scalar[:width]})
            print(f"verify {trace['name']} M={width}: {statistics.median(samples):.2f} ms", flush=True)
            save()
    rows = report["baseline_rows"]
    report["summary"] = {"baseline_mean_tps": statistics.mean(r["tok_per_s_decode"] for r in rows),
                         "baseline_max_tps": max(r["tok_per_s_decode"] for r in rows),
                         "all_linux_token_identity": all(r["token_identity_to_linux"] for r in rows)}
    save()
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
