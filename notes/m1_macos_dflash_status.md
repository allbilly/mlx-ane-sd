# M1 macOS DFlash: native full-ANE speedup measured

The latest experiment follows the M4 full-ANE offload method with the pinned
public Qwen3-0.6B pair: two 14-layer LUT6 target chunks, an incremental draft
projector, a three-layer LUT6 draft, and ANE vocabulary projections in a native
Swift/Core ML runner. Four prompts × two passes × 100 tokens measured:

| Configuration | Mean decode tok/s |
| --- | ---: |
| Unchanged MLX bf16 Metal baseline | 45.19 |
| Same LUT6 ANE target, padded eight-row greedy control | 53.36 |
| Full-ANE LUT6 DFlash SD | 72.00 |

**1.59× versus Metal (ratio of means), 1.35× versus the same ANE target's padded
eight-row AR control; best paired trial 2.13×. The 3× goal was not reproduced.**
An optimized one-row ANE autoregressive baseline has not been measured. All 800 SD token
IDs match greedy decoding through the same compressed target. LUT6 changes
output versus the original bf16 target, so this is a quantized 0.6B adaptation,
not an exact reproduction of the M4 Pro/64 GB Qwen3-4B result.

All five compiled artifacts assign major compute to ANE. Independent IOReport
hardware counters recorded 18,061 MB of ANE reads during the two-second native
activity probe versus 16 MB idle. The 24 completed trials recorded 4,728 Core ML
predictions. Real-input draft checks and Swift/Python token parity passed on
all four prompts. The initial unpooled native run failed with an IOSurface
allocation error; it is retained separately and excluded from the result.

See [the full-ANE report and reproduction commands](m1_m4_recipe.md),
[raw receipt](m1_m4_recipe.json), and [audit](m1_m4_recipe_audit.json).
Original Asahi source files and the archived capture kit remain unchanged.

## Earlier GPU control and ANE-draft-only experiments

Metal access was restored after the user allowed unrestricted execution. The
sweep completed all six configurations under one exclusive device reservation.
The GPU-only control did **not reproduce 3×** on this M1/8 GB setup. This
control did not use ANE and cannot establish the ANE speedup ceiling.

The separate native ANE draft now executes through ANEForge with e5rt device
mask `0x4`. Its initial FP16 RMSNorm overflow was fixed with power-of-two
rescaling, and every configuration passed real-input parity on four prompts.
The completed 24-trial run recorded 1,946 native ANE evaluations. The ANE body
and ANE vocabulary head with serial GPU verification matched all 800 token IDs,
averaging 16.47 tok/s against a paired stock baseline of 26.09 tok/s:
**0.66× mean paired speedup**, **1.02× best trial**. The 3× result was not
reproduced with this smaller model and an MLX target. Baseline rates varied
across the desktop session; this is not a hardware speedup ceiling.

See the [native ANE report](m1_macos_ane_dflash.md),
[audit](m1_macos_ane_dflash_audit.json), and
[reference survey](m1_macos_ane_reference_survey.md). Initial zero-output runs
remain marked invalid in `m1_macos_ane_dflash_zero_output.*`; the superseded
unwired control is retained in `m1_macos_ane_dflash_unwired.*`.

The GPU-only configuration matching all 800 token IDs averaged **26.20 tok/s**, against
**45.98 tok/s** for stock greedy MLX: **0.57× mean**, **0.61× best trial**.
The fastest experimental variant reached 33.88 tok/s, **0.73× mean**, but changed
tokens on math and story. All seven cache/kernel checks passed. The source
hashes, trial counts, token lengths, and greedy-reference identities were checked
against the raw receipts after completion.

See the [full report](m1_macos_dflash_sweep.md),
[summary JSON](m1_macos_dflash_sweep.json), and
[restored device visibility](m1_macos_metal_visibility_restored.json).

## Earlier launch failure

The earlier sweep waited for exclusive GPU/ANE access as requested. After the
terminal execution environment changed, its process was no longer present and
it had written no benchmark receipts. A fresh launch failed before acquiring
device locks or loading model weights:

```text
[blocked] MLX cannot access Metal: [metal::load_device] No Metal device available.
This typically occurs in headless, sandboxed, or virtualized macOS sessions
where the GPU is not accessible.
```

The structured failure record is [m1_macos_dflash_launch.json](m1_macos_dflash_launch.json).
At that time, the terminal had a filesystem sandbox and no escalation capability.
Waiting for another workload to release its locks cannot resolve this separate
Metal-access failure.

A follow-up probe confirmed native arm64 macOS, the expected MLX environment,
and an active Apple M1 GPU driver in IORegistry. Calling Apple's Metal framework
directly, without importing MLX, returned zero devices from `MTLCopyAllDevices`
and a null device from `MTLCreateSystemDefaultDevice`. The result is saved in
[m1_macos_metal_visibility.json](m1_macos_metal_visibility.json). This narrows
the failure to GPU visibility in that execution session; the sandbox was
the likely cause. [MLX's device loader](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp)
raises this exception when both native discovery methods return no device.

## Experiment

Hardware: base M1 MacBook Air, 8 GB. The local target is the pinned
`mlx-community/Qwen3-0.6B-bf16` checkpoint, revision
`42096995f6402fde107068cf530136fe64b604f8`. The public trained draft is
[`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`](https://huggingface.co/orestis-z/dflash-qwen3-0.6b-microcycle-dflash),
revision `4dd1e04078f993593338ef1f9403179e41e4580e`. Both are already downloaded.
The target stays bf16; one variant changes only the draft to fp16.

The driver measures all four repository prompts with two passes and up to 100
new tokens. It sweeps one, three and seven draft proposals, then two experimental
small-block Metal variants. If batched verification changes the greedy output,
it also measures a serial verification reference. Each generation uses fresh
caches; timing includes drafting, target verification, context updates and
rollback. A result qualifies only if its tokens match both the scalar target
loop and stock greedy mlx-lm. It reports mean and maximum speedup and retains
all failed numerical variants.

This uses MLX on the GPU. It does not reproduce the README's Qwen3-4B full-ANE
stack on an M4 Pro with 64 GB. The earlier isolated Metal-versus-Vulkan speed
comparison also does not establish an end-to-end SD speedup.

## Launch from a macOS terminal with Metal access

```bash
cd /Users/yeren/Desktop/mlx-ane-sd
.asahi/venv-metal/bin/python -u scripts/run_macos_dflash_sweep.py \
  > notes/m1_macos_dflash_sweep.log 2>&1
```

The runner checks Metal before waiting, then reserves the existing `~/ane.lock`
and `~/gpu.lock` for the complete sweep. It keeps waiting if either device is
reserved by another session. It reads existing lock files without changing their
contents. Results go to `notes/m1_macos_dflash_sweep*.json`; the final report is
`notes/m1_macos_dflash_sweep.md`.

## Verification status

The restored terminal passed all seven deterministic correctness checks covering
cache commits, rollback, partial/full acceptance, EOS, and the experimental
Metal kernel before timing. All five new Python files also compiled successfully.
The earlier failed launch remains recorded separately from benchmark results.

No existing Asahi source files were changed. The archived macOS capture kit
remains intact.
