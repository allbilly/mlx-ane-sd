# Asahi SD: project survey and measured bottlenecks

Update, 2026-10-06: the matched M1 macOS captures are now available. The compact
five-program full-ANE stack replays on Linux with all 588 output hashes matching.
After native useful-row readback and process scheduling changes, the four-pass
confirmation averages **61.39 versus 36.22 tok/s MLX (1.695×)**, with all 1,600
SD tokens matching macOS. See [the current results](m1_asahi_full_ane_results.md)
and [the confirmed recipe and controls](m1_asahi_scheduler_results.md).
The survey and per-operation measurements below describe the earlier path;
its Vulkan route diagnostics remain relevant to GPU verification follow-ups.

2026-10-05. All 18 nonhidden top-level Git checkouts under `~/` were inspected.
Other repositories were read only; copied sources, diagnostics and generated
artifacts are in `mlx-ane-sd`. Revisions, origins and observed working-tree
status are recorded in [home_project_inventory.json](asahi/home_project_inventory.json).

The available projects supply useful parts, but none supplies the original
macOS full-ANE Qwen3/DFlash stack for this Linux runtime. **Matched M1 captures
are useful for missing fused ANE graphs and coefficient layouts. They also
need verified replay and do not fix GPU verification or draft acceptance.**

## What can be reused

| Project | Useful contribution | Limit for this experiment |
|---|---|---|
| `ane` | M1 driver/UAPI, H13G HWX parser, verified GPT-2 fused-graph replay and training fixtures | GPT-2's 768-wide, head-64 schedules differ from Qwen3's width 1024 and head-128 attention. Its recent Qwen3.5 work has different cache semantics. |
| `Orion` | Private macOS ANE runtime, IOSurface I/O, MIL builders, fused graph and coefficient-packing patterns | macOS runtime. Relevant MIT components are copied into `asahi/vendor/Orion`; the local helper retains evaluated artifacts. |
| `mil-hwx-compiler` | Independent Linux H13G compilation and packing for 49 captured GPT-2 graphs | The executable needs ICU76; a local extracted runtime repaired loading. All five prepared SD graph types then failed `h13g.legalize.unsupported-graph`. |
| `joshuaswarren_mil-hwx-compiler` | Broader H13/H14 primitive encoders, manifests, host reference and Linux runner | Source inspection shows attention/chain geometry limits. Its fused FFN envelope is rows=375, 1024→4096→1024 with biases; this draft uses rows=32, 1024→3072→1024 and a gated FFN. No complete DFlash/Qwen3 backend was found. This checkout was not built/run here. |
| `qwen3.c` | Resident FP16 ANE linear plans, batch-32 prefill, native CPU target and lock coordination | Already vendored here with the FP16 API. Its newer `fp16/` work targets RK3588, not Apple ANE. Separate Q8 CPU rates are a different precision/workload. |
| `omarchy-mlx` | Vulkan target, dispatch diagnostics, single-token GEMV and larger-matrix kernels, ANE exporter | Small verify blocks use different routes. Its latest B>1 RoPE/norm fix concerns multiple requests; our target requests have B=1. It does not remove sequence-width reduction differences. |
| `ane-linux-experiments` | Converter fixes, derived task bindings, same-die Linux/macOS comparisons and negative optimization receipts | Shapes, chip, OS/compiler and driver ABI must match. Receipts are evidence of their workloads, not this SD speedup. |
| `ane-linux-experiments-qwen-ane-inference` | Projection, normalization, attention and staged-decode validation scripts | Same repository revision with additional working-tree files. Useful geometry/binding examples; not the trained dense Qwen3 DFlash loop here. |
| `coreml_to_ane_hwx` | macOS CoreML/Espresso→HWX export and format analysis | Export alone is not execution. Modern MIL/stateful graphs and compressed coefficient bindings still require decoding and replay validation. |
| `applegpu` | M1 AGX capture/replay examples and Mesa/native ISA comparisons | Verified ADD/MUL/render examples, not a transformer GEMM/attention library. Useful to inspect shader/command overhead; no ready SD kernel. |
| `coreglass` | Workload capture, per-phase summaries and provenance for missing counters | Profiling tool. Stock driver counters cannot substitute for measured ANE/GPU busy time. Direct timings here are userspace elapsed spans. |
| `old_whisper.cpp` | Historical `anecc`/HWX conversion and libane integration | Older model/container conventions. Its advertised path does not establish compatibility with current compiled SD graphs. |
| `linux` | Kernel/DRM source for submission, memory mapping and clock analysis | Source checkout is not evidence that a particular change is active in the running kernel. |
| `llama.cpp-mirai-s` | Vulkan attention/matmul implementation ideas and a separate performance reference | Different model/runtime/precision contracts. Porting a kernel into MLX requires matching its arithmetic and bindings. |
| `uzu` | Traceable model implementations and CPU numerical reference patterns | Primarily a different Rust/macOS stack. Mirai/Qwen3.5 cache behavior does not fit this trimmable dense-Qwen3 SD loop. |
| `calm` | Fused projections, activation kernels and bandwidth-focused inference design | CUDA implementation; no directly runnable M1 Vulkan/ANE backend. |
| `yalm` | Explicit phase boundaries, roofline and kernel benchmark patterns | CUDA/CPU implementation; CUDA kernels cannot execute on M1. |
| `mlx-ane-sd` | Original macOS conversion/runner plus this Linux trained-draft experiment | Original headline uses a different target/draft and M4 Pro 64 GB. Current Linux receipts use an M1 8 GB and a small 0.6B pair. |

## GPU verification is already enough to erase the gain

The previous four-prompt, two-pass, 100-token batch SD run averaged 20.59 tok/s
for its paired bf16 baseline and 9.22 tok/s for SD: mean paired speedup 0.45×,
maximum trial 0.58×. Four of eight outputs diverged. The lossless serial verifier
averaged 22.30→15.56 tok/s, mean 0.70× and maximum 0.75×, with 800/800 tokens
matching. See [the full experiment](asahi_dflash_findings.md).

The batch run emitted 2.106 tokens/cycle and spent 200.3 ms/cycle in target
verification. **Even a zero-cost draft would yield only about 10.5 emitted
tok/s at that acceptance and verification cost.** Fixing ANE drafting alone
cannot make this configuration beat its 20.59 tok/s baseline. Its measured
~251 ms total cycle needs roughly 5.2 tokens/cycle to break even; observed
acceptance is near two.

New isolated target probes used the same four real teacher-forced prefixes,
fresh equal histories, one warmup and three measured calls. Times include
cache evaluation, argmax and transfer of the three target feature captures.
These are diagnostic call latencies, not an end-to-end SD speed claim.

Cells show **mean / maximum prompt median**, in milliseconds.

| Query width | stock Vulkan | forced batched GEMV | linear inputs padded to 32 |
|---:|---:|---:|---:|
| 1 | 48.0 / 50.2 | scalar route retained | scalar route retained |
| 2 | 166.1 / 169.2 | 67.3 / 70.4 | 181.9 / 184.0 |
| 4 | 162.6 / 166.4 | 104.7 / 108.1 | 180.6 / 184.0 |
| 8 | 163.7 / 167.9 | 175.9 / 178.5 | 179.6 / 185.2 |
| 16 | 159.5 / 163.9 | 304.3 / 313.1 | 176.0 / 178.2 |
| 32 | 164.7 / 168.2 | stock retained | stock retained |

The installed fork uses specialized single-token BF16 GEMV, generic small-M
BF16 GEMM, and faster larger-matrix routes whose gate starts at M=32. Attention
also switches from fused single-query decode to composed batched evaluation.
This explains why a two-token call can cost much more than a one-token call.
The source and actual dispatch trace agree; these costs are measured.

Simply reshaping inputs to `[M,1,K]` is insufficient: a rank-2 weight operand
lets MLX flatten the operation back into GEMM. Explicit stride-zero broadcasting
of the RHS preserves the batched GEMV route. That route matched scalar outputs
bit for bit for all 16 tested real-input projection cases, including the full
vocabulary head. Stock/padded routes differed in every projection case.
Nevertheless, batched GEMV is slower at block 8, and padding does not improve
the full target. Full-model attention can still differ. This is a diagnostic
route, not a lossless SD optimization.

At these fixed prefixes, all routes matched scalar token decisions for widths
up to 16; stock width 32 matched three of four prompts. This limited check does
not supersede the failed 100-token batch SD identity gate.

Raw evidence: [target/linear timings](asahi/sd_routes.json),
[dispatch trace](asahi/row_gemv_dispatch.log), and
[aggregated timing data](asahi/sd_bottleneck_summary.json).
The retained [first reshape trial](asahi/sd_routes_implicit_batch.json)
used an implicit RHS and was flattened by MLX; its label did not actually
select GEMV. Its original receipt is preserved, and the corrected probe is
the one summarized above.

## The draft is not one compiled ANE graph

Clock instrumentation was added to the copied C backend behind `ANE_PROFILE=1`.
Normal execution leaves it disabled. Four real prefixes, three warmed repeated
proposals each, gave:

| Portion | mean ms/proposal |
|---|---:|
| Input packing/FP16 conversion | 1.98 |
| Upload and submission setup | 0.69 |
| Synchronous submission ioctl | 17.23 |
| Download | 1.03 |
| Output unpacking/finite checks | 3.93 |
| Outside the native calls | 15.00 |
| **Whole proposal** | **39.85** |

The maximum measured proposal was 46.47 ms across the twelve samples.

There are **59 submissions/proposal**: 21 body projections and 38 head chunks.
Head submission spans account for 13.03 ms; body spans account for 4.20 ms.
The ioctl spans include ANE compute and driver wait; they are not a measurement
of pure driver overhead. The remaining time includes NumPy attention/norms,
Python/ctypes coordination, conversions and argmax. The native program computes
32 physical rows while only 8 body rows and 7 head rows are useful.

The macOS stack uses fused transformer graphs, compressed coefficients,
dedicated heads and target chunks. Replacing that with individual FP16
projections changes compilation, data movement, useful batch work and placement.
The measurements do not support blaming a single poorly written kernel.

Raw evidence: [native profile](asahi/dflash_native_profile.json). Instrumentation
is diagnostic and adds overhead; the existing throughput receipts remain the
end-to-end measurements. Five FP16 hardware shape/guard checks passed after
adding the profile API in [this receipt](asahi/ane_profile_extension_check.json).
An additional full-draft CPU comparison attempt waited for another workload's
ANE lock and reached its 180-second deadline; it did not run. That attempt is
retained in `asahi/dflash_profile_extension_check.log`. Earlier full-draft
comparisons are listed in the original findings; they use the earlier binary.

## What to capture on this M1

The available Linux H13G compiler was tested with real-weight RMSNorm, fused
gated FFN, cached attention and both head sizes. All five parsed and were
rejected with `h13g.legalize.unsupported-graph`. See
[the compiler receipt](asahi/m1_compiler_coverage.json). The initial missing
ICU76 loader failure is retained separately; the final receipt uses an extracted
Fedora runtime under this project's `.asahi/`, without changing system libraries
or the compiler repository.

A prepared kit at `.asahi/m1-sd-capture-ready/` contains **20 real-input ANE
cases**, the pinned bf16 target and a matched Metal target probe. Runtime and
tools have provenance and 142 verified file hashes. The kit preserves evaluated
runtime artifacts, outputs, a separately compiled H13G HWX, coefficient files,
macOS/compiler metadata and timings. Offline HWX and evaluated runtime payloads
must be compared before claiming they represent the same program.

Follow [the macOS capture instructions](../asahi/macos/README.md). It requires
this same base M1 booted into macOS. Preparation/hash checks and shell/Python
syntax checks passed here. **Mac execution and Linux replay of the new cases
are pending.** The captured outputs will let us validate replay, bindings and
coefficient packing instead of inferring them from unrelated shapes.

The first kit captures FP16 graph fusion. Subsequent work still needs
compressed/LUT6 coefficient support, mutable K/V state and whole-draft/target
chunks. M4/H16G objects are not directly usable by this H13G driver/parser.
The matched Metal probe distinguishes same-chip GPU behavior from the original
M4 Pro comparison. A better accepted draft, efficient verification, and the
complete ANE execution path remain necessary to reproduce a useful SD gain.

## Reproduce the Linux diagnostics

```bash
cd ~/mlx-ane-sd
bash asahi/build.sh
timeout 600 scripts/asahi_python.sh scripts/probe_asahi_sd_routes.py
timeout 180 scripts/asahi_python.sh scripts/profile_asahi_dflash.py
```

Both diagnostics acquire `~/ane.lock` then `~/gpu.lock`. Do not overlap hardware
workloads. Source/binary/model hashes are retained in the receipts. Fresh-cache
full SD reproduction remains documented in `notes/asahi_dflash_findings.md`.
