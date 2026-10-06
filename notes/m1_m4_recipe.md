# M1 macOS: native Swift full-ANE DFlash

The M4 full-ANE offload method reaches **72.00 tok/s** on this M1, versus **45.19 tok/s** for stock MLX bf16 (**1.59× ratio of means**, 1.60× mean paired speedup). SD is 1.35× versus the same LUT6 ANE target's **padded eight-row AR control**. An optimized one-row ANE autoregressive baseline has not been measured.

Hardware: base M1 MacBook Air, 8 GB, macOS 27.0.1. This adapts the method to the already downloaded, pinned public `mlx-community/Qwen3-0.6B-bf16` target and `orestis-z/dflash-qwen3-0.6b-microcycle-dflash` draft. It is **not an exact reproduction of Qwen3-4B on M4 Pro/64 GB**, and cannot establish an M1-versus-M4 hardware ceiling. The HF CLI dry-run put the original bf16 4B checkpoint at about 8 GB, exceeding the free space available.

## Method

The standalone Swift/Core ML runner uses two 14-layer LUT6 target chunks, captures target hidden states at layers 1/13/25, an incremental draft-context projector, a three-layer LUT6 draft, and a shared LUT6 vocabulary head. The target and head use per-grouped-channel palettization (group 16); the draft uses per-tensor palettization. The full 151,936-token vocabulary is split into nineteen 8,192-column tiles, without restricting candidate tokens. External caches commit only accepted positions. The trained block size is eight; at most seven tokens are proposed per cycle.

Core ML performs target and draft compute, final normalization, and both vocabulary projections with `cpuAndNeuralEngine`. CPU performs embedding lookup, cache copies, argmax and acceptance. MLX is used only by the unchanged bf16 Metal baseline and validation reference. RMSNorm rescales before squaring to prevent FP16 overflow on real draft context values.

Four repository prompts × two passes × up to 100 new tokens. Trial order reverses in the second pass. Both existing GPU/ANE locks are reserved. Compilation, loading and prefill are excluded; drafting, target verification, projections, copies and rejection handling are included. The native timer includes the first post-prefill forward; stock mlx-lm's generation timing differs by roughly one token at this generation length. The ANE autoregressive control uses the same fixed eight-row target graph with unused rows padded; it is a numerical/control reference, not an optimized one-row ANE autoregressive implementation.

| Prompt | MLX bf16 tok/s | Padded ANE AR tok/s | ANE LUT6 SD tok/s | SD / MLX | SD / padded AR |
| --- | ---: | ---: | ---: | ---: | ---: |
| capital | 45.02 | 50.82 | 74.39 | 1.65× | 1.46× |
| fibonacci | 44.10 | 52.64 | 85.33 | 1.93× | 1.62× |
| math | 46.36 | 56.69 | 72.62 | 1.57× | 1.28× |
| story | 45.27 | 53.27 | 55.67 | 1.23× | 1.05× |
| **Mean** | **45.19** | **53.36** | **72.00** | **1.59×** | **1.35×** |

Best paired trial versus MLX: 2.13×. A best trial is not the four-prompt mean.

## Correctness and hardware evidence

SD matches ordinary decoding through the same compressed ANE target in **8/8 trials**. Compared with the original bf16 MLX target, **0/8 trials** match. LUT6 changes target numerics; any drift is a quantization trade-off, not a byte-identical bf16 reproduction.

Real-input draft checks passed on all four prompts (minimum hidden-state cosine 0.998388 versus the fp16 reference with the same target features). Swift and Python Core ML output identical token IDs on all four short parity runs. Every compiled artifact assigns its major compute operations to ANE. Compute-plan placement is a compilation estimate; the independent IOReport hardware byte counters provide runtime activity evidence.

The native command-line runner drains an autorelease pool after each prediction and request so Core ML's temporary IOSurfaces are released. The initial unpooled long run failed with an E5RT IOSurface allocation error; its incomplete receipts are retained in `m1_m4_recipe_unpooled.*` and excluded from this result.

Over two seconds, idle ANE reads were **16.31 MB**, versus **18061.03 MB** during the native full-stack activity probe. The benchmark records **4,728 Core ML predictions** in its measured ANE trials. The earlier one-layer probe also independently demonstrated positive runtime ANE traffic.

## Reproduce

```bash
cd /Users/yeren/Desktop/mlx-ane-sd
# Use a fresh directory: the original cache predates fingerprint receipts.
.asahi/venv-metal/bin/python scripts/convert_macos_m4_recipe.py \
  --out .asahi/m4-recipe-m1-reviewed
swiftc -O -framework CoreML swift-bench/m1_full_ane.swift \
  -o .asahi/m4-recipe-m1-reviewed/m1-full-ane
clang -fobjc-arc -framework Foundation \
  ~/ane-llm-measurements/tools/ane_counters/anebytes.m \
  -o .asahi/m4-recipe-m1-reviewed/anebytes
.asahi/venv-metal/bin/python scripts/bench_macos_m4_recipe.py --native \
  --artifacts .asahi/m4-recipe-m1-reviewed --out notes/m1_m4_recipe_reviewed.json
.asahi/venv-metal/bin/python scripts/report_macos_m4_recipe.py \
  --receipt notes/m1_m4_recipe_reviewed.json \
  --report notes/m1_m4_recipe_reviewed.md --audit notes/m1_m4_recipe_reviewed_audit.json
```

The environment helper reads NumPy/Torch/Core ML from the existing `~/more-ane-transformers/.venv` while using current MLX/tokenizers from the workspace venv. It installs no dependencies and modifies no reference repositories. Original Asahi files and the existing capture archive are unchanged.

[Raw trials, traces and placement](m1_m4_recipe.json), [audit](m1_m4_recipe_audit.json), [one-layer hardware probe](m1_m4_recipe_probe_runtime.json).

The original measured sources were preserved before the review fixes. This audit verifies their recorded hashes against the snapshots; the raw measurements have not been rewritten or rerun with the revised converter and benchmark gates.
