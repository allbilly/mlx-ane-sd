"""Audit the completed full-ANE M1 receipts and write the measured result."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from macos_m4_benchmark_checks import summarize, require_same_target_identity


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_source(root, receipt_sha, name, expected):
    relative=Path("swift-bench" if name.endswith(".swift") else "scripts")/name
    current=root/relative
    if current.is_file() and digest(current)==expected:
        return str(relative)
    archived=root/"notes/m1_m4_recipe_sources"/receipt_sha/relative
    if not archived.is_file() or digest(archived)!=expected:
        raise ValueError(f"Measured source hash does not match current or preserved source: {name}")
    return str(archived.relative_to(root))


def main(argv=None):
    root=Path(__file__).resolve().parent.parent
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt",type=Path,default=root/"notes/m1_m4_recipe.json")
    ap.add_argument("--report",type=Path,default=root/"notes/m1_m4_recipe.md")
    ap.add_argument("--audit",type=Path,default=root/"notes/m1_m4_recipe_audit.json")
    args=ap.parse_args(argv)
    path=args.receipt
    r=json.loads(path.read_text());assert r["status"]=="complete"
    identity=require_same_target_identity(r["rows"],r["repeats"])
    receipt_sha=digest(path)
    source_paths={}
    for name,expected in r["source_sha256"].items():
        source_paths[name]=verified_source(root,receipt_sha,name,expected)
    for name,expected in r["manifest"]["source_sha256"].items():
        source_paths[name]=verified_source(root,receipt_sha,name,expected)
    artifacts_root=Path(r["manifest"]["artifacts"][0]["compiled"]).parent.parent
    assert digest(artifacts_root/"m1-full-ane")==r["native_binary_sha256"]
    assert r["target_weights_sha256"]=="bfe05e58f5acdecd38e5f3e64c82071682683a73db0e57fd437523885a49f2ff"
    assert r["manifest"]["math_all_ane"]
    assert len(r["validation"])==len(r["native_parity"])==4
    assert all(v["identical"] for v in r["native_parity"])
    assert all(v["finite"] and v["nonzero"] and v["draft_cosine"]>.9 for v in r["validation"])
    assert len(r["rows"])==4*r["repeats"]*3
    for row in r["rows"]:
        assert 0<len(row["tokens"])<=r["max_new"]
        if row["impl"]=="mlx_bf16": continue
        assert len(row["trace"])==row["cycles"]
        flattened=[]
        for t in row["trace"]:
            prefix=0
            for a,b in zip(t["candidates"],t["predictions"]):
                if a!=b: break
                prefix+=1
            assert prefix==t["accepted"]
            expected=t["candidates"][:prefix]+[t["predictions"][prefix]]
            assert t["committed"]==expected[:len(t["committed"])]
            flattened.extend(t["committed"])
        assert flattened==row["tokens"][1:]
        for role in ("target0","target14","target_head"):
            assert row["calls"][role]==row["cycles"]+1
        if row["impl"]=="ane_lut6_sd":
            proposals=sum(bool(t["candidates"]) for t in row["trace"])
            assert row["calls"].get("draft",0)==row["calls"].get("draft_head",0)==proposals
            assert row["calls"]["projector"]==row["cycles"]+1
        assert sum(row["profile_s"].values())<=row["decode_s"]*1.01
    s=summarize(r["rows"]);assert s==r["summary"]
    idle=float(re.search(r"ANE_RD_MB ([0-9.]+)",r["idle_counter"])[1])
    active=float(re.search(r"ANE_RD_MB ([0-9.]+)",r["active_counter"])[1])
    assert active>10*max(1,idle),"No convincing active/idle ANE traffic difference"
    rate_ratio=s["ane_lut6_sd"]["mean_tok_s"]/s["mlx_bf16"]["mean_tok_s"]
    sd_ratio=s["ane_lut6_sd"]["mean_tok_s"]/s["ane_lut6_ar"]["mean_tok_s"]
    quality=s["tokens_vs_mlx"];internal=s["tokens_vs_ane_ar"]
    qualified=internal["identical_trials"]==internal["trials"]
    calls=sum(sum(row.get("calls",{}).values()) for row in r["rows"])
    lines=["# M1 macOS: native Swift full-ANE DFlash", "",
           f"The M4 full-ANE offload method reaches **{s['ane_lut6_sd']['mean_tok_s']:.2f} tok/s** "
           f"on this M1, versus **{s['mlx_bf16']['mean_tok_s']:.2f} tok/s** for stock MLX bf16 "
           f"(**{rate_ratio:.2f}× ratio of means**, {s['vs_mlx']['mean_paired_speedup']:.2f}× mean paired speedup). "
           f"SD is {sd_ratio:.2f}× versus the same LUT6 ANE target's **padded eight-row AR control**. "
           "An optimized one-row ANE autoregressive baseline has not been measured.", "",
           "Hardware: base M1 MacBook Air, 8 GB, macOS 27.0.1. This adapts the method to the already downloaded, "
           "pinned public `mlx-community/Qwen3-0.6B-bf16` target and "
           "`orestis-z/dflash-qwen3-0.6b-microcycle-dflash` draft. It is **not an exact reproduction of "
           "Qwen3-4B on M4 Pro/64 GB**, and cannot establish an M1-versus-M4 hardware ceiling. "
           "The HF CLI dry-run put the original bf16 4B checkpoint at about 8 GB, exceeding the free space available.", "",
           "## Method", "",
           "The standalone Swift/Core ML runner uses two 14-layer LUT6 target chunks, captures target hidden "
           "states at layers 1/13/25, an incremental draft-context projector, a three-layer LUT6 draft, and a "
           "shared LUT6 vocabulary head. The target and head use per-grouped-channel palettization (group 16); "
           "the draft uses per-tensor palettization. The full 151,936-token vocabulary is split into nineteen "
           "8,192-column tiles, without restricting candidate tokens. External caches commit only accepted "
           "positions. The trained block size is eight; at most seven tokens are proposed per cycle.", "",
           "Core ML performs target and draft compute, final normalization, and both vocabulary projections "
           "with `cpuAndNeuralEngine`. CPU performs embedding lookup, cache copies, argmax and acceptance. "
           "MLX is used only by the unchanged bf16 Metal baseline and validation reference. "
           "RMSNorm rescales before squaring to prevent FP16 overflow on real draft context values.", "",
           "Four repository prompts × two passes × up to 100 new tokens. Trial order reverses in the second "
           "pass. Both existing GPU/ANE locks are reserved. Compilation, loading and prefill are excluded; "
           "drafting, target verification, projections, copies and rejection handling are included. "
           "The native timer includes the first post-prefill forward; stock mlx-lm's generation timing "
           "differs by roughly one token at this generation length. The ANE autoregressive control uses "
           "the same fixed eight-row target graph with unused rows padded; it is a numerical/control "
           "reference, not an optimized one-row ANE autoregressive implementation.", "",
           "| Prompt | MLX bf16 tok/s | Padded ANE AR tok/s | ANE LUT6 SD tok/s | SD / MLX | SD / padded AR |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in s["mlx_bf16"]["per_prompt"]:
        a=s["mlx_bf16"]["per_prompt"][name];b=s["ane_lut6_ar"]["per_prompt"][name];c=s["ane_lut6_sd"]["per_prompt"][name]
        lines.append(f"| {name} | {a:.2f} | {b:.2f} | {c:.2f} | {c/a:.2f}× | {c/b:.2f}× |")
    lines.append(f"| **Mean** | **{s['mlx_bf16']['mean_tok_s']:.2f}** | **{s['ane_lut6_ar']['mean_tok_s']:.2f}** | "
                 f"**{s['ane_lut6_sd']['mean_tok_s']:.2f}** | **{rate_ratio:.2f}×** | **{sd_ratio:.2f}×** |")
    lines += ["",f"Best paired trial versus MLX: {s['vs_mlx']['max_trial_speedup']:.2f}×. "
              "A best trial is not the four-prompt mean.", "", "## Correctness and hardware evidence", "",
              f"SD matches ordinary decoding through the same compressed ANE target in "
              f"**{internal['identical_trials']}/{internal['trials']} trials**. Compared with the original "
              f"bf16 MLX target, **{quality['identical_trials']}/{quality['trials']} trials** match. "
              "LUT6 changes target numerics; any drift is a quantization trade-off, not a byte-identical "
              "bf16 reproduction.", "",
              f"Real-input draft checks passed on all four prompts (minimum hidden-state cosine "
              f"{min(v['draft_cosine'] for v in r['validation']):.6f} versus the fp16 reference with the same "
              "target features). Swift and Python Core ML output identical token IDs on all four short "
              "parity runs. Every compiled artifact assigns its major compute operations to ANE. "
              "Compute-plan placement is a compilation estimate; the independent IOReport hardware "
              "byte counters provide runtime activity evidence.", "",
              "The native command-line runner drains an autorelease pool after each prediction and "
              "request so Core ML's temporary IOSurfaces are released. The initial unpooled long run "
              "failed with an E5RT IOSurface allocation error; its incomplete receipts are retained "
              "in `m1_m4_recipe_unpooled.*` and excluded from this result.", "",
              f"Over two seconds, idle ANE reads were **{idle:.2f} MB**, versus **{active:.2f} MB** during "
              f"the native full-stack activity probe. The benchmark records **{calls:,} Core ML predictions** "
              "in its measured ANE trials. The earlier one-layer probe also independently demonstrated "
              "positive runtime ANE traffic.", "", "## Reproduce", "", "```bash",
              "cd /Users/yeren/Desktop/mlx-ane-sd",
              "# Use a fresh directory: the original cache predates fingerprint receipts.",
              ".asahi/venv-metal/bin/python scripts/convert_macos_m4_recipe.py \\",
              "  --out .asahi/m4-recipe-m1-reviewed",
              "swiftc -O -framework CoreML swift-bench/m1_full_ane.swift \\",
              "  -o .asahi/m4-recipe-m1-reviewed/m1-full-ane",
              "clang -fobjc-arc -framework Foundation \\",
              "  ~/ane-llm-measurements/tools/ane_counters/anebytes.m \\",
              "  -o .asahi/m4-recipe-m1-reviewed/anebytes",
              ".asahi/venv-metal/bin/python scripts/bench_macos_m4_recipe.py --native \\",
              "  --artifacts .asahi/m4-recipe-m1-reviewed --out notes/m1_m4_recipe_reviewed.json",
              ".asahi/venv-metal/bin/python scripts/report_macos_m4_recipe.py \\",
              "  --receipt notes/m1_m4_recipe_reviewed.json \\",
              "  --report notes/m1_m4_recipe_reviewed.md --audit notes/m1_m4_recipe_reviewed_audit.json", "```", "",
              "The environment helper reads NumPy/Torch/Core ML from the existing "
              "`~/more-ane-transformers/.venv` while using current MLX/tokenizers from the workspace venv. "
              "It installs no dependencies and modifies no reference repositories. Original Asahi files "
              "and the existing capture archive are unchanged.", "",
              f"[Raw trials, traces and placement]({path.name}), "
              f"[audit]({args.audit.name}), "
              "[one-layer hardware probe](m1_m4_recipe_probe_runtime.json).", ""]
    archived={n:p for n,p in source_paths.items() if p.startswith("notes/m1_m4_recipe_sources/")}
    if archived:
        lines += ["The original measured sources were preserved before the review fixes. "
                  "This audit verifies their recorded hashes against the snapshots; the raw measurements "
                  "have not been rewritten or rerun with the revised converter and benchmark gates.", ""]
    args.report.write_text("\n".join(lines))
    audit={"receipt_sha256":receipt_sha,"source_hashes_match":True,
           "verified_source_paths":source_paths,"same_target_check":identity,"trials":len(r["rows"]),
           "coreml_predictions":calls,"all_math_planned_ane":True,"active_ane_read_mb":active,
           "idle_ane_read_mb":idle,"native_parity_trials":4,"sd_matches_same_target":qualified,
           "speedup_ratio_of_means":rate_ratio,"speedup_vs_padded_ane_ar_ratio_of_means":sd_ratio,
           "optimized_one_row_ane_ar_measured":False,
           "speedup_3x_mean_reproduced":qualified and rate_ratio>=3,
           "exact_original_4b_reproduction":False}
    args.audit.write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__": main()
