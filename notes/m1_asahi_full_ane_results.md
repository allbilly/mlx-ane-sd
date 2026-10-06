# M1 Asahi full-ANE replay and benchmark — 2026-10-06

**The full stack replays correctly on Linux. After host transport and CPU
scheduling changes, SD averages 1.695× the stock MLX/Vulkan baseline.** All five reconstructed H13G programs execute through
the installed ANE driver; all 588 selected outputs match the macOS goldens
byte-for-byte, and all four complete SD generation traces match.

The latest four-prompt, four-pass confirmation gives **61.39 tok/s SD versus
36.22 tok/s MLX BF16 (1.695×)**, and 51.94 tok/s for the same LUT6 ANE AR
control. All 1,600 SD tokens and acceptance/cache decisions match macOS. Native
useful-row readback, P-core affinity and a per-process utilization hint close
the relative speedup gap. See [the current recipe, controls and receipts](m1_asahi_scheduler_results.md).
The earlier resident-cache stage gave 39.50 / 35.19 tok/s (1.123×); the initial
replay was 26.96 / 34.72 tok/s (0.776×).
The matched M1 macOS result is **72.00 versus 45.19 tok/s (1.593×)**;
macOS did not measure the 0.78× slowdown. These are separate baselines on
the same chip, using Python/DRM/Vulkan and Swift/CoreML/Metal respectively.

## Setup and verification

- Base M1 MacBook Air, T8103, 8 GB; Fedora Asahi Remix 44, kernel `7.1.13+`.
- Reference/source commit `9a824a5`; compact kit from `artifacts/m1-full-ane/`.
- Pinned Qwen3-0.6B BF16 target and microcycle DFlash checkpoints, already
  cached here. Every complete reconstructed HWX hash matches the reference.
- Two 14-layer LUT6 target chunks, LUT6 draft/projector, shared complete
  vocabulary head; B=8 and C=256. The runtime cache occupies 501,871,902 bytes.
- `mlx-omarchy 0.32.4.dev202610041653+6edd258`, `mlx-lm 0.31.3`, NumPy 2.5.3.
  Installed fork provenance is `match`; see
  [the provenance receipt](asahi/m1_full_ane_mlx_provenance_20261006.json).
- Compact-kit hashes, all five buffer plans, and 16 host tests passed.

The driver was installed but unloaded after boot. The user loaded it and
granted this session a device ACL. The initial verification attempt stopped
before submitting any programs because the device was inaccessible; its
[failed receipt](m1_linux_full_ane_verify_20261006.json) is preserved.

The next attempt passed **72 selected calls / 588 output hashes**, covering
all four prompts and early/decode/populated-cache stages. The verifier also
regenerated every 100-token SD trace, including acceptance and cache offsets.
This establishes output equivalence of the offline HWX under Linux replay
with the macOS CoreML goldens. It does not identify the binary loaded internally
by CoreML on macOS.

[Successful verification receipt](m1_linux_full_ane_verify_20261006_access.json),
[preflight and reconstruction](asahi/m1_full_ane_preflight_20261006.json).

## Earlier benchmark after resident-cache transport changes

The replay now reuses compiler-layout input storage and checks FP16 finiteness
through exponent bits. Target and draft K/V buffers remain resident; only the
accepted cache prefix is written after each forward. Reset clears both the CPU
mirror and the device cache. Proposals do not modify committed state, so rejected
suffixes never enter the cache.

Vocabulary argmax reads requested rows directly from mapped outputs and converts
them to FP32 for reduction. Numeric chunk ordering and strict `>` comparisons
preserve the lowest token ID on ties. Verification still reads every physical
output and compares mapped reductions with full-readback token choices.

Both changes passed all **72 calls / 588 output hashes**, all four complete
100-token traces, and **386 full-versus-mapped head reduction comparisons**.
An additional verification hashes the actual device input mappings against the
goldens, confirming that resident cache surfaces match the regenerated CPU
inputs, including padding and populated-cache positions.
Host tests cover all FP16 bit patterns, padded/strided layouts, partial cache
commit/reset, invalid updates and vocabulary ties across more than ten chunks.

| Configuration | Mean tok/s | Max trial tok/s |
|---|---:|---:|
| Stock MLX BF16 / Vulkan | 35.19 | 36.23 |
| LUT6 ANE AR, padded B=8 | 30.61 | 38.89 |
| LUT6 ANE SD | 39.50 | 46.13 |

| Prompt | MLX BF16 | ANE AR | ANE SD | SD / MLX |
|---|---:|---:|---:|---:|
| capital | 35.37 | 36.62 | 45.26 | 1.280× |
| fibonacci | 33.91 | 33.52 | 41.03 | 1.210× |
| math | 35.50 | 27.50 | 41.20 | 1.161× |
| story | 35.97 | 24.82 | 30.51 | 0.848× |

The ratio of mean SD/MLX rates is **1.12268×**, mean paired speedup
**1.12514×**, maximum paired trial **1.33666×**. SD/ANE AR is **1.29032×**
by mean rates. SD throughput improves **46.5%** over the initial transport
implementation. Story still loses to MLX; acceptance and phase costs vary
substantially across prompts.

The protocol remains four prompts × two passes × 100 tokens, with 12 excluded
warmups and fresh caches. All eight SD trials / 800 tokens match the compressed
ANE AR target and captured macOS generations. The two math trials match BF16;
BF16 byte identity is not claimed for the other prompts.

[Optimized verification](m1_linux_full_ane_verify_resident_surfaces_20261006.json),
[optimized benchmark](m1_linux_full_ane_bench_resident_mapped_20261006.json),
[derived summary](asahi/m1_full_ane_resident_mapped_summary_20261006.json).

## Initial paired benchmark

Four prompts × two passes × 100 generated tokens, fresh caches, alternating
configuration order. All three configurations warmed every prompt for ten
tokens before measured trials: 12 warmup calls excluded from the 24 rows.
Both `~/ane.lock` and `~/gpu.lock` were held throughout.

| Configuration | Mean tok/s | Max trial tok/s |
|---|---:|---:|
| Stock MLX BF16 / Vulkan | 34.72 | 35.95 |
| LUT6 ANE AR, padded B=8 | 17.59 | 18.22 |
| LUT6 ANE SD | 26.96 | 30.73 |

| Prompt | MLX BF16 | ANE AR | ANE SD | SD / MLX |
|---|---:|---:|---:|---:|
| capital | 35.79 | 17.89 | 29.26 | 0.818× |
| fibonacci | 35.04 | 17.27 | 30.06 | 0.858× |
| math | 34.10 | 18.02 | 26.97 | 0.791× |
| story | 33.96 | 17.19 | 21.54 | 0.634× |

The SD/MLX ratio of mean rates is **0.77638×**; mean paired speedup is
**0.77543×**, maximum paired trial **0.86007×**. SD/ANE AR is **1.53223×** by
mean rates, with maximum paired trial **1.74491×**.

All eight SD trials / 800 tokens match the same LUT6 ANE AR target and the
captured macOS compressed-target generations. Only the two math trials match
the local BF16 MLX baseline. LUT6 changes target numerics; this is not a
byte-identical BF16 acceleration result. The AR control uses padded B=8
kernels; an optimized B=1 ANE baseline remains unmeasured.

The stock MLX rate uses `stream_generate.generation_tps`, excluding time to the
first token. ANE host decode timing includes the last prompt-token forward.
These are the conventions retained from the macOS benchmark; prefill, loading
and warmup are excluded. The macOS native Swift result remains 72.00 tok/s SD,
53.36 tok/s ANE AR and 45.19 tok/s Metal BF16. Python/Linux and Swift/CoreML
have different host execution paths.

[Raw benchmark](m1_linux_full_ane_bench_20261006.json),
[derived summary](asahi/m1_full_ane_summary_20261006.json).

## Initial profile: where Linux time went

The benchmark's SD program-call means are 20.48 ms for target chunk 0,
20.33 ms for chunk 14, 15.39 ms for target head, 15.46 ms for draft head,
4.02 ms for draft body and 0.85 ms for projector. These spans include CPU
transport and synchronous submission, not just ANE computation.

An additional diagnostic wraps the existing replay calls and measures packing,
mapped writes/reads, unpacking and the submission ioctl separately. It runs
all four complete real SD traces and confirms unchanged tokens and acceptance
traces. Its statistics include prefill and decode calls and are separate from
the uninstrumented benchmark rates.

| Program | Mean total ms/call | Pack | Mapped read | Synchronous submit | Calls |
|---|---:|---:|---:|---:|---:|
| target chunk 0 | 19.65 | 9.72 | 2.16 | 6.64 | 216 |
| target chunk 14 | 19.49 | 9.64 | 2.09 | 6.63 | 216 |
| projector | 0.81 | 0.067 | 0.425 | 0.203 | 216 |
| draft body | 3.85 | 2.05 | 0.071 | 1.50 | 190 |
| shared head | 14.94 | 0.024 | 10.37 | 2.70 | 386 |

Other measured portions include mapped writes, unpacking, scratch initialization
and output initialization. Complete means/maxima and per-prompt call times are
in [the profile receipt](asahi/m1_full_ane_profile_20261006.json). The submit
span includes device execution and driver wait; it cannot be interpreted as
pure compute or pure driver overhead.

Each target chunk packs/uploads **14,705,152 input bytes per forward**, mainly
the entire fixed-capacity K/V cache. The pair therefore repeats approximately
29.4 MB of input preparation even when only one or a few positions change.
Input mapped writes themselves average approximately 0.56 ms per chunk; the
larger measured cost is inside `pack_tensor`, which includes finite checking,
layout conversion and allocation/copies. The head reads **2,430,976 bytes** of
logits for all eight physical rows on every call; mapped readback alone costs
about 10.37 ms.

These measurements motivated the validated transport changes above:

1. Separate finite checking from layout/copy costs, reuse compiler-layout input
   buffers, and update only accepted cache positions. Preserve zero padding,
   reset and rollback semantics; verify the existing goldens after changes.
2. Read vocabulary logits only for the useful rows, or reduce argmax while
   reading the mapped output. Retain a complete-output verification path;
   avoid materializing padded rows during measured generation.
3. The paired benchmark has been repeated after verification. Additional macOS
   dumps are unnecessary for the currently verified five-program computation.
   Live driver state remains useful for explaining a remaining submission gap.

## Profile after transport changes and remaining work

The additional profile reproduces all four complete SD traces. Like the initial
profile, it includes prefill plus decode and is separate from throughput trials.

| Program | Mean total ms/call | Input preparation | Mapped read | Synchronous submit | Calls |
|---|---:|---:|---:|---:|---:|
| target chunk 0 | 9.15 | 0.047 | 2.20 | 6.66 | 216 |
| target chunk 14 | 9.08 | 0.042 | 2.14 | 6.66 | 216 |
| projector | 0.76 | 0.042 | 0.432 | 0.208 | 216 |
| draft body | 1.72 | 0.039 | 0.072 | 1.49 | 190 |
| shared head | 13.64 | 0.015 | included below | 2.71 | 386 |

Each target forward now uploads **25,088 bytes**, versus 14,705,152 previously.
Accepted-prefix updates are measured separately: about 0.42 / 0.40 ms per
target chunk and 0.11 ms for the draft. Average update writes are 143,891 bytes
per target and 30,834 per draft, including the larger prefill commits.

Head reduction still costs **10.79 ms/call**, including direct mapped reads,
FP16→FP32 conversion, finite checks and argmax. Avoiding the intermediate full
copy does not eliminate expensive memory access. The local driver source maps
BOs with `pgprot_writecombine`; this is a candidate explanation for slow CPU
reads, not a measured attribution of the entire 10.79 ms to the mapping type.
No driver or other repository was modified.

[Optimized profile, including cache-update costs](asahi/m1_full_ane_profile_resident_mapped_20261006.json).

Further performance work has three concrete targets:

These tasks were identified after the resident-cache stage. Items 1 and 2
are now implemented and verified in [the current results](m1_asahi_scheduler_results.md).

1. Reduce vocabulary bytes consumed. At that stage, target verification read all
   candidate rows even after the first rejection makes later decisions unused.
   An early-stop reduction or an M1-compiled argmax head could reduce readback;
   preserve emitted tokens, accepted prefixes and cache state when validating it.
2. Reduce target K/V output readback to committed positions and compare cached
   staging/native copy approaches. At that stage, two target chunks spent ~4.3 ms total
   reading complete physical outputs. Any driver mapping change needs a defined
   CPU/device coherency protocol and its own measurements.
3. Obtain a true B=1 ANE AR control. All five supplied programs are compiled for
   B=8; reading one useful row does not turn them into single-row kernels.
   New M1 compilation/captures are required. Small-block Vulkan optimization
   remains a separate path with existing route and numerical diagnostics.

Current draft emission is only 1.7–2.4 tokens/cycle across the four prompts.
A stronger accepted draft can help, but it is separate from the demonstrated
46.5% throughput improvement from host transport. Exact macOS-loaded HWX and
live driver register traces remain unavailable; Linux output equivalence is
already established by the golden checks.

## Reproduction on this Asahi checkout

The module must be loaded and the current session must have device access.
Use fresh receipt paths; existing evidence is not overwritten.

For the confirmed 1.695× result, use [the optimized reproduction command](m1_asahi_scheduler_results.md#reproduce-the-one-worker-result),
including native/prefix readback, P-core affinity, the utilization hint and
100-token warmups. The default transport is `--transport resident --head-readback mapped`.
`--transport reference --head-readback full` provides full-cache packing and
complete-output readback for comparison; `packed` and `rows` are intermediate
controls. The reference control also benefits from reused scratch/poison bytes
and exponent-bit output checks, so it does not promise the original timings.
Full physical golden verification runs under every option.

```bash
replay_target=/home/asahi/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/42096995f6402fde107068cf530136fe64b604f8

scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode verify --target "$replay_target" \
  --draft .asahi/models/microcycle-dflash --out notes/m1_linux_full_ane_verify_new.json

scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode bench --target "$replay_target" \
  --draft .asahi/models/microcycle-dflash --max-new 100 --repeats 2 \
  --out notes/m1_linux_full_ane_bench_new.json

scripts/asahi_python.sh scripts/profile_m1_full_ane_replay.py \
  artifacts/m1-full-ane --target "$replay_target" \
  --out notes/asahi/m1_full_ane_profile_new.json
```

Runtime HWX files remain in ignored `.asahi/m1-full-ane-cache/`; learned
coefficients are not added to Git. Other source repositories were only read.
