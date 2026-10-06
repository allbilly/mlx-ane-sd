"""Run and summarize the M1 3x DFlash attempt under one device reservation."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import bench_macos_dflash as bench


def summarize(path):
    data = json.loads(path.read_text())
    rows = data["rows"]
    summary = dict(data["summary"])
    summary["receipt"] = str(path.relative_to(bench.REPO))
    summary["identity_trials"] = sum(r["token_identity"] for r in rows)
    summary["stock_identity_trials"] = sum(r["stock_token_identity"] for r in rows)
    summary["generated_tokens"] = sum(len(r["dflash"]["tokens"]) for r in rows)
    summary["trials"] = len(rows)
    summary["verification"] = data["verification"]
    summary["max_speedup_vs_stock"] = max(r["speedup_vs_stock"] for r in rows)
    summary["qualified"] = summary["all_token_identity"] and summary["all_stock_token_identity"]
    summary["tokens_per_cycle"] = statistics.mean(r["dflash"]["tokens_per_cycle"] for r in rows)
    summary["mean_phase_ms_per_cycle"] = {
        "verify": statistics.mean(r["dflash"]["target_verify_s"] / r["dflash"]["cycles"] * 1000 for r in rows),
        **{phase: statistics.mean(r["dflash"]["draft_profile_s"][phase] / r["dflash"]["cycles"] * 1000
                                  for r in rows) for phase in ("context", "body", "head")}}
    # Measured ceiling if draft and context work took zero time, at the same
    # observed acceptance. This is a diagnostic bound, not a measured SD rate.
    summary["zero_draft_cost_speedup_bound_vs_scalar"] = statistics.mean(
        r["dflash"]["tokens_per_cycle"] * r["baseline"]["decode_s"] / len(r["baseline"]["tokens"])
        / (r["dflash"]["target_verify_s"] / r["dflash"]["cycles"]) for r in rows)
    summary["zero_draft_cost_speedup_bound_vs_stock"] = statistics.mean(
        r["dflash"]["tokens_per_cycle"]
        / (r["dflash"]["target_verify_s"] / r["dflash"]["cycles"])
        / r["stock_mlx_lm"]["generation_tps"] for r in rows)
    summary["prompts"] = []
    for name, _ in bench.PROMPTS:
        selected = [r for r in rows if r["name"] == name]
        summary["prompts"].append({"name": name,
            "baseline_tps": statistics.mean(r["stock_mlx_lm"]["generation_tps"] for r in selected),
            "dflash_tps": statistics.mean(r["dflash"]["tok_per_s_decode"] for r in selected),
            "speedup": statistics.mean(r["speedup_vs_stock"] for r in selected),
            "max_speedup": max(r["speedup_vs_stock"] for r in selected),
            "tokens_per_cycle": statistics.mean(r["dflash"]["tokens_per_cycle"] for r in selected),
            "identity": all(r["token_identity"] and r["stock_token_identity"] for r in selected)})
    return summary


def write_report(result, path):
    configs = result["configs"]
    qualified = [c for c in configs if c["qualified"]]
    best = max(qualified, key=lambda c: c["mean_speedup_vs_stock"]) if qualified else None
    result["best_qualified_config"] = best["name"] if best else None
    result["reproduced_3x_mean"] = bool(best and best["mean_speedup_vs_stock"] >= 3)
    result["best_qualified_mean_speedup"] = best["mean_speedup_vs_stock"] if best else None
    result.setdefault("finished_utc", datetime.now(timezone.utc).isoformat())
    path.write_text(json.dumps(result, indent=2) + "\n")
    report = path.with_suffix(".md")
    lines = ["# M1 macOS DFlash 3× reproduction attempt", "",
             "Machine: base Apple M1 MacBook Air, 8 GB. Target remains the pinned "
             "`mlx-community/Qwen3-0.6B-bf16`; public trained draft: "
             "[`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`]"
             "(https://huggingface.co/orestis-z/dflash-qwen3-0.6b-microcycle-dflash).", "",
             f"Four unchanged repo prompts × {result['repeats']} alternating-order passes × "
             f"up to {result['max_new']} tokens. "
             "Each generation starts with fresh caches. Draft features remain resident in MLX. "
             "Decode time includes drafting, context updates, verification and cache rollback. "
             "Stock mlx-lm uses greedy sampling and the same 32-token prefill chunks. "
             "Stock generation_tps excludes the first token; the custom loop includes its forward. "
             "The separate scalar-loop baseline and both token sequences are retained in every receipt.", "",
             "A configuration qualifies only when all SD tokens equal scalar greedy decoding "
             "and that baseline also equals stock mlx-lm. Reported speedups below use stock mlx-lm.", "",
             "| Configuration | Stock baseline tok/s | SD tok/s | Mean paired speedup | Max trial | SD identity | Stock identity | Qualified |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for c in configs:
        lines.append(f"| {c['name']} | {c['stock_mean_tps']:.2f} | {c['dflash_mean_tps']:.2f} | "
                     f"{c['mean_speedup_vs_stock']:.2f}× | {c['max_speedup_vs_stock']:.2f}× | "
                     f"{c['identity_trials']}/{c['trials']} | {c['stock_identity_trials']}/{c['trials']} | "
                     f"{'yes' if c['qualified'] else 'no'} |")
    lines += ["", "**3× mean reproduced:** " + ("yes" if result["reproduced_3x_mean"] else "no") + ".", ""]
    if best:
        lines += [f"Best configuration passing all token checks: **{best['name']}**, "
                  f"**{best['mean_speedup_vs_stock']:.2f}× mean**, "
                  f"**{best['max_speedup_vs_stock']:.2f}× best trial** versus stock mlx-lm. "
                  f"All {best['generated_tokens']} generated token IDs match both greedy references.", "",
                  "| Prompt | Stock tok/s | SD tok/s | Paired mean | Max trial | Tokens/cycle |",
                  "|---|---:|---:|---:|---:|---:|"]
        for p in best["prompts"]:
            lines.append(f"| {p['name']} | {p['baseline_tps']:.2f} | {p['dflash_tps']:.2f} | "
                         f"{p['speedup']:.2f}× | {p['max_speedup']:.2f}× | {p['tokens_per_cycle']:.2f} |")
    lines += ["", "## Measured bottleneck", "",
              "| Configuration | Tokens/cycle | Verify ms/cycle | Context ms/cycle | Draft body ms/cycle | Draft head ms/cycle | Zero-draft-cost bound vs stock |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for c in configs:
        phase = c["mean_phase_ms_per_cycle"]
        lines.append(f"| {c['name']} | {c['tokens_per_cycle']:.2f} | {phase['verify']:.2f} | "
                     f"{phase['context']:.2f} | {phase['body']:.2f} | {phase['head']:.2f} | "
                     f"{c['zero_draft_cost_speedup_bound_vs_stock']:.2f}× |")
    batch = [c for c in configs if c["verification"] == "batch"]
    verify_times = [c["mean_phase_ms_per_cycle"]["verify"] for c in batch]
    draft_times = [sum(v for k, v in c["mean_phase_ms_per_cycle"].items() if k != "verify")
                   for c in configs]
    stock_ms = 1000 / statistics.mean(c["stock_mean_tps"] for c in configs)
    lines += ["", f"The draft commits {min(c['tokens_per_cycle'] for c in configs):.2f}–"
              f"{max(c['tokens_per_cycle'] for c in configs):.2f} tokens per cycle across configurations. "
              f"Batched verification alone takes {min(verify_times):.0f}–{max(verify_times):.0f} ms per cycle, "
              f"versus about {stock_ms:.0f} ms per token for stock greedy MLX. "
              f"Drafting and context updates add {min(draft_times):.0f}–{max(draft_times):.0f} ms per cycle. "
              "The diagnostic bound holds the observed acceptance and verification timings fixed "
              "and removes all other cycle work; it is not a measured speedup or a universal hardware ceiling. "
              f"Its highest mean value is {max(c['zero_draft_cost_speedup_bound_vs_stock'] for c in configs):.2f}×.", "",
              "The sweep runs all seven cache/kernel checks before timing. "
              f"It retained {sum(c['trials'] for c in configs)} paired trials. "
              "The identity columns distinguish SD-versus-scalar checks from scalar-versus-stock checks."]
    serial = [c for c in configs if c["verification"] == "serial" and c["qualified"]]
    if serial and any(not c["qualified"] for c in batch):
        lines += ["", "Serial verification matched both greedy references on every trial. "
                  "The failures with batched verification isolate the token drift to batched target "
                  "numerics for this checkpoint, rather than rejected-cache rollback."]
    lines += ["", "The optional small-block Metal kernel keeps bf16 target weights and activations "
              "and accumulates dot products in FP32. Its reduction order differs from stock MLX; "
              "passing the token checks is required. fp16 variants change only the drafter. "
              "They still verify with the bf16 target. Failed numerical variants are retained "
              "and excluded from the qualified best result.", "",
              "This smaller-model M1 experiment is separate from the README's M4 Pro/64 GB "
              "Qwen3-4B full-ANE stack. The 4B reproduction recorded about 10 GB peak memory. "
              "These runs measure MLX GPU drafting and verification, including an experimental "
              "Metal kernel; they do not establish a full-ANE result on M1.", "",
              "Reproduction (reuse the downloaded, pinned model files):", "", "```bash",
              ".asahi/venv-metal/bin/python -u scripts/run_macos_dflash_sweep.py", "```", "",
              "Model downloads, when needed, use the Hugging Face CLI:", "", "```bash",
              "hf download mlx-community/Qwen3-0.6B-bf16 \\",
              "  --revision 42096995f6402fde107068cf530136fe64b604f8 \\",
              "  --local-dir .asahi/models/qwen3-0.6b-bf16-metal",
              "hf download orestis-z/dflash-qwen3-0.6b-microcycle-dflash \\",
              "  --revision 4dd1e04078f993593338ef1f9403179e41e4580e \\",
              "  --local-dir .asahi/models/microcycle-dflash --include '*.json' '*.safetensors'",
              "python3 - <<'PY'",
              "import json",
              "from pathlib import Path",
              "Path('.asahi/models/microcycle-dflash/SOURCE.json').write_text(json.dumps({",
              "    'model': 'orestis-z/dflash-qwen3-0.6b-microcycle-dflash',",
              "    'revision': '4dd1e04078f993593338ef1f9403179e41e4580e'}, indent=2) + '\\n')",
              "PY",
              ".asahi/venv-metal/bin/python scripts/run_macos_dflash_sweep.py \\",
              "  --target .asahi/models/qwen3-0.6b-bf16-metal", "```", "",
              "Receipts:", ""]
    lines += [f"- [{c['name']}]({Path(c['receipt']).name})" for c in configs]
    lines += ["", "All existing Asahi source files and the archived capture kit remain unchanged.", ""]
    report.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=Path, default=bench.REPO / ".asahi/m1-sd-macos/target")
    ap.add_argument("--draft", type=Path, default=bench.REPO / ".asahi/models/microcycle-dflash")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=Path, default=bench.REPO / "notes/m1_macos_dflash_sweep.json")
    args = ap.parse_args()
    if args.max_new < 2 or args.repeats < 1:
        ap.error("Need max-new>=2 and repeats>=1")
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "max_new": args.max_new,
              "repeats": args.repeats, "configs": []}
    try:
        bench.require_metal()
    except RuntimeError as error:
        result.update(status="blocked", reason=str(error),
                      checked_utc=datetime.now(timezone.utc).isoformat())
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[blocked] {error}", flush=True)
        return 1
    result["status"] = "waiting_for_devices"
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    configs = [("p7_bf16_stock", 7, "bf16", "none", "batch"),
               ("p3_bf16_stock", 3, "bf16", "none", "batch"),
               ("p1_bf16_stock", 1, "bf16", "none", "batch"),
               ("p7_bf16_small_metal", 7, "bf16", "both", "batch"),
               ("p7_fp16_draft_small_metal", 7, "fp16", "both", "batch")]
    with bench.device_locks():
        result["status"] = "running"
        for i, (name, proposals, precision, kernel, verification) in enumerate(configs):
            receipt = args.out.with_name(f"{args.out.stem}_{name}.json")
            command = ["--target", str(args.target), "--draft", str(args.draft),
                       "--max-new", str(args.max_new), "--repeats", str(args.repeats),
                       "--proposals", str(proposals), "--draft-precision", precision,
                       "--small-matmul", kernel, "--verify", verification, "--stock-baseline",
                       "--out", str(receipt)]
            if i == 0:
                command.append("--check-cache")
            print(f"[configuration] {name}", flush=True)
            status = bench.main(command)
            if status not in (0, 2):
                raise RuntimeError(f"Benchmark failed: {name}, status {status}")
            entry = summarize(receipt)
            entry["name"] = name
            result["configs"].append(entry)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            # A scalar verification run is required when a batched variant
            # diverges, to separate draft compatibility from verifier numerics.
            if verification != "serial" and i == len(configs) - 1 and any(
                    not c["qualified"] for c in result["configs"]):
                configs.append(("p7_bf16_serial_reference", 7, "bf16", "none", "serial"))
    result["status"] = "complete"
    write_report(result, args.out)
    print(json.dumps({k: result[k] for k in ("best_qualified_config", "best_qualified_mean_speedup",
                                           "reproduced_3x_mean")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
