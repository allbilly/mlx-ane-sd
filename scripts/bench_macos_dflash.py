"""Paired macOS DFlash benchmark, with fresh caches and greedy token checks.

Decode timing includes the last prompt-token forward, resident draft context
updates, drafting, target verification and rejection rollback. All four repo
prompts are measured; repeat order alternates baseline/SD. No cached baseline
tokens or hidden states are used by the measured drafter.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_final_stack import PROMPTS

REPO = Path(__file__).resolve().parent.parent
_LOCK_DEPTH = 0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_state():
    state = {}
    for name, command in (("swap", ["sysctl", "vm.swapusage"]),
                          ("thermal", ["pmset", "-g", "therm"])):
        probe = subprocess.run(command, text=True, capture_output=True)
        state[name] = {"returncode": probe.returncode, "stdout": probe.stdout.strip(),
                       "stderr": probe.stderr.strip()}
    return state


def require_metal():
    # Check before waiting for device locks: a restricted terminal can import
    # stdlib successfully while MLX cannot access the Metal device at all.
    try:
        import mlx.core as mx
    except ImportError as error:
        raise RuntimeError(f"MLX cannot access Metal: {error}") from error
    if not mx.metal.is_available():
        raise RuntimeError("Native Metal MLX required")
    return mx


@contextlib.contextmanager
def device_locks():
    # A synchronous sweep can hold the reservation across several main() calls.
    global _LOCK_DEPTH
    if _LOCK_DEPTH:
        _LOCK_DEPTH += 1
        try:
            yield
        finally:
            _LOCK_DEPTH -= 1
        return
    with contextlib.ExitStack() as stack:
        for name in ("ane.lock", "gpu.lock"):
            path = Path.home() / name
            try:
                stream = path.open("r")
            except FileNotFoundError:
                stream = path.open("a")
            lock = stack.enter_context(stream)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[wait] another workload holds {name}", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
        _LOCK_DEPTH = 1
        try:
            yield
        finally:
            _LOCK_DEPTH = 0


def target_forward(model, ids, cache, feature_ids=(), logits=True):
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    h = model.model.embed_tokens(mx.array(ids, mx.uint32)[None])
    mask = create_attention_mask(h, cache[0])
    captured = {}
    for i, (layer, c) in enumerate(zip(model.model.layers, cache), 1):
        h = layer(h, mask, c)
        if i in feature_ids:
            captured[i] = h
    features = mx.concatenate([captured[i] for i in feature_ids], axis=-1)[0] if feature_ids else None
    if logits:
        h = model.model.norm(h)
        scores = model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
        tokens = mx.argmax(scores, axis=-1)[0]
    else:
        tokens = None
    mx.eval([c.state for c in cache], *([tokens] if tokens is not None else []),
            *([features] if features is not None else []))
    return tokens.tolist() if tokens is not None else None, features


def generate(model, tokenizer, draft, prompt, limit, speculative, verification="batch"):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    if limit < 1:
        raise ValueError("Token limit must be positive")
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("Empty tokenized prompt")
    cache = make_prompt_cache(model)
    captures = draft.feature_ids if speculative else ()
    if speculative:
        draft.reset()
    mx.synchronize()
    start = time.perf_counter()
    for begin in range(0, len(ids) - 1, 32):
        _, features = target_forward(model, ids[begin:min(begin + 32, len(ids) - 1)], cache, captures, False)
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
            raise RuntimeError("Draft/target cache offset mismatch")
        candidates = draft.propose(tokens[-1])[0] if limit - len(tokens) > 1 else []
        candidates = candidates[:max(0, limit - len(tokens) - 1)]
        begin = time.perf_counter()
        if verification == "serial":
            predictions, features_parts = [], []
            for i, token in enumerate([tokens[-1], *candidates]):
                prediction, feature = target_forward(model, [token], cache, captures)
                predictions.extend(prediction)
                features_parts.append(feature)
                if i < len(candidates) and prediction[0] != candidates[i]:
                    break
            features = mx.concatenate(features_parts)
        else:
            predictions, features = target_forward(model, [tokens[-1], *candidates], cache, captures)
        verify_s += time.perf_counter() - begin
        count = 0
        for candidate, prediction in zip(candidates, predictions):
            if candidate != prediction:
                break
            count += 1
        committed = [*candidates[:count], predictions[count]]
        for i, token in enumerate(committed):
            if token in tokenizer.eos_token_ids:
                committed = committed[:i + 1]
                break
        keep = len(committed)
        trim_prompt_cache(cache, len(predictions) - keep)
        draft.append(features[:keep])
        if any(c.offset != position + keep for c in cache) or draft.offset != position + keep:
            raise RuntimeError("Cache did not roll back rejected positions")
        tokens.extend(committed)
        traces.append({"position": position, "proposals": candidates, "target_predictions": predictions,
                       "accepted_prefix": count, "committed": committed})
        accepted += min(count, keep)
        proposed += len(candidates)
        cycles += 1
    mx.synchronize()
    elapsed = time.perf_counter() - start
    return {"tokens": tokens, "text": tokenizer.decode(tokens), "prompt_tokens": len(ids),
            "prefill_s": prefill_s, "decode_s": elapsed, "tok_per_s_decode": len(tokens) / elapsed,
            "cycles": cycles, "accepted": accepted, "proposed": proposed,
            "acceptance_rate": accepted / proposed if proposed else 0,
            "tokens_per_cycle": (len(tokens) - 1) / cycles if cycles else None,
            "target_verify_s": verify_s, "verification_trace": traces,
            "draft_profile_s": dict(draft.profile) if speculative else None}


def stock_baseline(model, tokenizer, prompt, limit):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    last = None
    tokens = []
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=limit,
                                    sampler=make_sampler(temp=0), prefill_step_size=32):
        tokens.append(response.token)
        last = response
    return {"tokens": tokens, "generation_tps": last.generation_tps,
            "generation_tokens": last.generation_tokens}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=REPO / ".asahi/m1-sd-macos/target")
    ap.add_argument("--draft", type=Path, default=REPO / ".asahi/models/microcycle-dflash")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--proposals", type=int, default=7)
    ap.add_argument("--draft-precision", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--verify", choices=("batch", "serial"), default="batch")
    ap.add_argument("--small-matmul", choices=("none", "draft", "both"), default="none",
                    help="Experimental weight-reusing Metal kernel; baseline always uses stock kernels")
    ap.add_argument("--stock-baseline", action="store_true", help="Also check stock mlx-lm greedy output and speed")
    ap.add_argument("--check-cache", action="store_true", help="Run cache and Metal kernel checks while holding device locks")
    ap.add_argument("--out", type=Path, default=REPO / "notes/m1_macos_dflash.json")
    args = ap.parse_args(argv)
    if platform.system() != "Darwin":
        ap.error("This runner requires macOS")
    if args.max_new < 2 or args.repeats < 1:
        ap.error("Need max-new>=2 and repeats>=1")
    mx = require_metal()
    from mlx_lm import load
    from dflash_mlx_speculators import DFlash
    result = {"schema": 1, "started_utc": datetime.now(timezone.utc).isoformat(),
              "machine": mx.device_info(), "macos": platform.mac_ver()[0],
              "packages": {p: importlib.metadata.version(p) for p in ("mlx", "mlx-lm")},
              "target": {"model": "mlx-community/Qwen3-0.6B-bf16",
                         "revision": "42096995f6402fde107068cf530136fe64b604f8", "path": str(args.target)},
              "draft": {"source": json.loads((args.draft / "SOURCE.json").read_text()),
                        "config": json.loads((args.draft / "config.json").read_text()),
                        "precision": args.draft_precision, "proposals": args.proposals},
              "sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), REPO / "scripts/dflash_mlx_speculators.py")},
              "max_new": args.max_new, "repeats": args.repeats, "verification": args.verify,
              "small_matmul": args.small_matmul,
              "procedure": __doc__, "rows": []}

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    save()
    with device_locks():
        if args.check_cache:
            import unittest
            checks = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromName("test_macos_dflash"))
            if not checks.wasSuccessful():
                raise RuntimeError("Cache or Metal kernel checks failed")
            result["checks"] = {"tests_run": checks.testsRun, "passed": True,
                                "test_source_sha256": sha256(REPO / "scripts/test_macos_dflash.py"),
                                "kernel_source_sha256": sha256(REPO / "scripts/macos_small_matmul.py")}
        print("[load] bf16 target + public Speculators DFlash, native Metal", flush=True)
        result["host_before"] = host_state()
        result["target"]["weights_sha256"] = sha256(args.target / "model.safetensors")
        if result["target"]["weights_sha256"] != "bfe05e58f5acdecd38e5f3e64c82071682683a73db0e57fd437523885a49f2ff":
            raise RuntimeError("Target weights differ from the pinned bf16 checkpoint")
        result["draft"]["weights_sha256"] = sha256(args.draft / "model.safetensors")
        model, tok = load(str(args.target))
        if model.model.embed_tokens.weight.dtype != mx.bfloat16:
            raise RuntimeError("Expected unchanged bf16 target weights")
        model.eval()
        draft = DFlash(args.draft, args.proposals, args.draft_precision,
                       shared_embedding=model.model.embed_tokens.weight)
        result["draft"]["shared_identical_embedding_head"] = draft.shared_weights
        if draft.config["speculators_config"]["verifier"]["name_or_path"] != "Qwen/Qwen3-0.6B":
            raise RuntimeError("Checkpoint does not match target")
        target_small = []
        from macos_small_matmul import install, set_enabled
        if args.small_matmul != "none":
            install(draft.network)
            result["sources_sha256"]["macos_small_matmul.py"] = hashlib.sha256(
                (REPO / "scripts/macos_small_matmul.py").read_bytes()).hexdigest()
        if args.small_matmul == "both":
            target_small = install(model)
        print("[warmup] both complete generation paths", flush=True)
        for spec in (False, True):
            set_enabled(target_small, spec)
            generate(model, tok, draft, "The weather is", 12, spec, args.verify)
        if args.stock_baseline:
            set_enabled(target_small, False)
            stock_baseline(model, tok, "The weather is", 12)
        for repeat in range(args.repeats):
            for name, prompt in PROMPTS:
                row = {"repeat": repeat, "name": name, "prompt": prompt}
                for spec in ((False, True) if repeat % 2 == 0 else (True, False)):
                    mode = "dflash" if spec else "baseline"
                    set_enabled(target_small, spec)
                    row[mode] = generate(model, tok, draft, prompt, args.max_new, spec, args.verify)
                    print(f"{repeat} {name} {mode}: {row[mode]['tok_per_s_decode']:.2f} tok/s", flush=True)
                row["token_identity"] = row["baseline"]["tokens"] == row["dflash"]["tokens"]
                row["speedup"] = row["dflash"]["tok_per_s_decode"] / row["baseline"]["tok_per_s_decode"]
                row["first_difference"] = next((i for i, pair in enumerate(zip(
                    row["baseline"]["tokens"], row["dflash"]["tokens"])) if pair[0] != pair[1]), None)
                if args.stock_baseline:
                    set_enabled(target_small, False)
                    row["stock_mlx_lm"] = stock_baseline(model, tok, prompt, args.max_new)
                    row["stock_token_identity"] = row["stock_mlx_lm"]["tokens"] == row["baseline"]["tokens"]
                    row["speedup_vs_stock"] = row["dflash"]["tok_per_s_decode"] / row["stock_mlx_lm"]["generation_tps"]
                result["rows"].append(row)
                save()
                print(f"  identity={row['token_identity']} speedup={row['speedup']:.3f}x "
                      f"tokens/cycle={row['dflash']['tokens_per_cycle']:.2f}", flush=True)
        rows = result["rows"]
        result["summary"] = {"baseline_mean_tps": statistics.mean(r["baseline"]["tok_per_s_decode"] for r in rows),
                             "dflash_mean_tps": statistics.mean(r["dflash"]["tok_per_s_decode"] for r in rows),
                             "mean_speedup": statistics.mean(r["speedup"] for r in rows),
                             "max_speedup": max(r["speedup"] for r in rows),
                             "all_token_identity": all(r["token_identity"] for r in rows),
                             "peak_mlx_memory_bytes": mx.get_peak_memory()}
        if args.stock_baseline:
            result["summary"].update(stock_mean_tps=statistics.mean(r["stock_mlx_lm"]["generation_tps"] for r in rows),
                                     mean_speedup_vs_stock=statistics.mean(r["speedup_vs_stock"] for r in rows),
                                     all_stock_token_identity=all(r["stock_token_identity"] for r in rows))
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["host_after"] = host_state()
        save()
        print(json.dumps(result["summary"], indent=2), flush=True)
    return 0 if result["summary"]["all_token_identity"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
