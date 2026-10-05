"""Offline Asahi Vulkan MLX baseline vs ordinary autoregressive speculation.

This small-model compatibility bench does not reproduce the macOS DFlash/ANE
stack. Both paths use a bf16 target; only the draft is quantized. Run under a
bounded timeout, with the mlx-omarchy wheel installed instead of upstream mlx.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_final_stack import PROMPTS

REPO = Path(__file__).resolve().parent.parent
TARGET = "mlx-community/Qwen3-0.6B-bf16"
TARGET_REV = "42096995f6402fde107068cf530136fe64b604f8"
DRAFT = "mlx-community/Qwen3-0.6B-4bit"
DRAFT_REV = "73e3e38d981303bc594367cd910ea6eb48349da8"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def memory_state():
    keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    return {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.split(":")[0] in keys
    }


def power_state():
    governors = {p.parent.name: p.read_text().strip() for p in
                 Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_governor")}
    temperatures = {str(p): p.read_text().strip() for p in
                    Path("/sys/class/thermal").glob("thermal_zone*/temp")}
    return {"governors": governors, "thermal_zone_temperatures": temperatures}


def model_snapshot(model, revision):
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, revision=revision, local_files_only=True))
    config = json.loads((path / "config.json").read_text())
    return path, {
        "model": model,
        "revision": path.name,
        "config": config,
        "files": {f.name: sha256(f) for f in sorted(path.iterdir())
                  if f.suffix in {".json", ".safetensors"}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--target-revision", default=TARGET_REV)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--draft-revision", default=DRAFT_REV)
    ap.add_argument("--num-draft", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--omarchy-repo", type=Path, default=REPO.parent / "omarchy-mlx")
    ap.add_argument("--out", type=Path, default=REPO / "notes/bench_asahi_mlx.json")
    args = ap.parse_args()
    if min(args.num_draft, args.max_new, args.repeats) < 1:
        ap.error("--num-draft, --max-new, and --repeats must be positive")
    if platform.system() != "Linux":
        ap.error("this harness records Linux hardware metadata")
    if any(os.environ.get(k) for k in (
        "MLX_OMARCHY_ALLOW_NON_APPLE", "MLX_OMARCHY_CAPS_SIM"
    )):
        ap.error("hardware measurements require the real Apple GPU capabilities")

    sys.path.insert(0, str(args.omarchy_repo / "scripts"))
    from mlx_provenance import installed_provenance, provenance_line

    prov = installed_provenance()
    print(provenance_line(prov), flush=True)
    if prov["verified"] != "match":
        raise RuntimeError(f"MLX binary provenance not verified: {prov}")

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step, speculative_generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler

    if mx.default_device() != mx.Device(mx.gpu):
        raise RuntimeError(f"expected GPU device, got {mx.default_device()}")
    info_bin = Path(mx.__file__).parent / "bin/mlx-omarchy-info"
    info = json.loads(subprocess.check_output([str(info_bin), "--json"], text=True))
    target_path, target_meta = model_snapshot(args.target, args.target_revision)
    draft_path, draft_meta = model_snapshot(args.draft, args.draft_revision)
    if target_meta["config"].get("quantization"):
        raise RuntimeError("this bench requires an unquantized bf16 target")
    if target_meta["config"]["model_type"] != "qwen3":
        raise RuntimeError("use dense Qwen3 with a trimmable cache")
    # Equal vocabulary sizes do not prove that token IDs have equal meanings.
    if target_meta["files"]["tokenizer.json"] != draft_meta["files"]["tokenizer.json"]:
        raise RuntimeError("target and draft tokenizer.json hashes differ")

    result = {
        "schema": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "harness_sha256": sha256(__file__),
        "kernel": platform.release(),
        "os_release": Path("/etc/os-release").read_text(),
        "machine": Path("/proc/device-tree/model").read_text().rstrip("\0"),
        "mlx_provenance": prov,
        "packages": {d: importlib.metadata.version(d) for d in (
            "mlx-omarchy", "mlx-lm", "transformers", "numpy", "huggingface-hub")},
        "vulkan": {k: v for k, v in info.items() if k not in {"ane", "coreml"}},
        "ane": info.get("ane"),
        "models": {"target": target_meta, "draft": draft_meta},
        "num_draft": args.num_draft,
        "max_new": args.max_new,
        "repeats": args.repeats,
        "procedure": "Raw prompts, greedy; fresh caches each run. Prefill all but "
                     "the last prompt token outside decode timing for both models. "
                     "Time every decode block, including the first. Stop at EOS or "
                     "max_new; token counts include EOS. Warm both paths for 8 "
                     "tokens; alternate baseline/SD order on successive repeats. "
                     "No governor changes or forced cooling.",
        "trace_enabled": "MLX_OMARCHY_TRACE_DISPATCH" in os.environ,
        "memory_before": memory_state(),
        "power_before": power_state(),
        "rows": [],
    }

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    save()
    print("[load] bf16 target and quantized draft", flush=True)
    target, tok = load(str(target_path))
    draft, _ = load(str(draft_path))
    if target.model.embed_tokens.weight.dtype != mx.bfloat16:
        raise RuntimeError("expected bf16 target weights")
    sampler = make_sampler(temp=0.0)

    def run(prompt, speculative, limit):
        ids = mx.array(tok.encode(prompt), mx.uint32)
        cache = make_prompt_cache(target)
        draft_cache = make_prompt_cache(draft) if speculative else []
        mx.synchronize()
        start = time.perf_counter()
        if ids.size > 1:
            target(ids[:-1][None], cache=cache)
            if speculative:
                draft(ids[:-1][None], cache=draft_cache)
            mx.eval([c.state for c in cache + draft_cache])
        mx.synchronize()
        prefill_s = time.perf_counter() - start
        gen = (speculative_generate_step(
            ids[-1:], target, draft, num_draft_tokens=args.num_draft,
            max_tokens=limit, sampler=sampler, prompt_cache=cache + draft_cache,
        ) if speculative else generate_step(
            ids[-1:], target, max_tokens=limit, sampler=sampler, prompt_cache=cache,
        ))
        tokens, flags = [], []
        start = time.perf_counter()
        try:
            for response in gen:
                token = int(response[0])
                tokens.append(token)
                flags.append(bool(response[2]) if speculative else False)
                if token in tok.eos_token_ids:
                    break
        finally:
            gen.close()
            mx.synchronize()
        decode_s = time.perf_counter() - start
        return {
            "tokens": tokens,
            "text": tok.decode(tokens),
            "prompt_tokens": ids.size,
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "tok_per_s_decode": len(tokens) / decode_s,
            "accepted_tokens": sum(flags),
            "draft_token_fraction": sum(flags) / len(tokens) if tokens else 0,
        }

    print("[warmup] both paths", flush=True)
    for spec in (False, True):
        run("Hello", spec, 8)
    for repeat in range(args.repeats):
        for name, prompt in PROMPTS:
            pair = {}
            for spec in ((False, True) if repeat % 2 == 0 else (True, False)):
                pair[spec] = run(prompt, spec, args.max_new)
            base, sd = pair[False], pair[True]
            mismatch = next((i for i, (b, s) in enumerate(zip(base["tokens"], sd["tokens"]))
                             if b != s), None)
            identical = base["tokens"] == sd["tokens"]
            if mismatch is None and not identical:
                mismatch = min(len(base["tokens"]), len(sd["tokens"]))
            row = {"repeat": repeat + 1, "name": name, "prompt": prompt,
                   "baseline": base, "speculative": sd,
                   "tokens_identical": identical, "first_mismatch_index": mismatch,
                   "speedup": sd["tok_per_s_decode"] / base["tok_per_s_decode"]}
            result["rows"].append(row)
            save()
            print(f"{repeat + 1} {name:10} baseline={base['tok_per_s_decode']:.2f} "
                  f"SD={sd['tok_per_s_decode']:.2f} tok/s "
                  f"speedup={row['speedup']:.2f}x identical={identical} "
                  f"draft fraction={sd['draft_token_fraction']:.1%}", flush=True)
    rows = result["rows"]
    result["summary"] = {
        "baseline_mean_tok_s": statistics.mean(r["baseline"]["tok_per_s_decode"] for r in rows),
        "sd_mean_tok_s": statistics.mean(r["speculative"]["tok_per_s_decode"] for r in rows),
        "mean_speedup": statistics.mean(r["speedup"] for r in rows),
        "max_speedup": max(r["speedup"] for r in rows),
        "all_tokens_identical": all(r["tokens_identical"] for r in rows),
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    result["memory_after"] = memory_state()
    result["power_after"] = power_state()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(result["summary"], indent=2), flush=True)
    if not result["summary"]["all_tokens_identical"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
