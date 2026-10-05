# Native ANE DFlash on Asahi

The trained DFlash draft now runs on the M1 ANE with a bf16 MLX/Vulkan target.
This reproduces the **block-diffusion drafting and verification mechanism** on
a smaller model. It does **not** reproduce the macOS 4B full-ANE 2.21× result.
The lossless reference is slower than greedy decoding; the batched bf16 path
also changes target decisions.

## Measured results

Host: base M1 MacBook Air, 8 GB, Fedora Asahi Remix 44. Target:
`mlx-community/Qwen3-0.6B-bf16`, revision
`42096995f6402fde107068cf530136fe64b604f8`. Draft:
[`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`](https://huggingface.co/orestis-z/dflash-qwen3-0.6b-microcycle-dflash/tree/4dd1e04078f993593338ef1f9403179e41e4580e),
revision `4dd1e04078f993593338ef1f9403179e41e4580e`.
It is a trained three-layer Qwen3 draft with an eight-token block and full
151936-token vocabulary. Its features use HF/vLLM hidden-state IDs 1, 13, 25,
meaning **after decoder layers 0, 12, 24**. These IDs differ from the
zero-based layer-output hooks used by the original z-lab MLX runner.

Four diverse raw prompts, 100 generated tokens each, two passes with reversed
baseline/SD order. Fresh caches, greedy decisions, one warmup of each path.
Decode timing includes the final prompt-token forward, all draft work,
verification, transfers, host decisions and cache updates. Model loading and
prefix prefill are excluded. Both paths use the same synchronous target
wrapper. This baseline differs from the pipelined `mlx-lm` baseline in the
earlier [MLX compatibility report](asahi_findings.md).

| bf16 verification | greedy mean tok/s | DFlash mean tok/s | mean paired speedup | best trial | token identity |
|---|---:|---:|---:|---:|---|
| scalar, stop at first rejection | 22.30 | 15.56 | 0.70× | 0.75× | all 8 runs / 800 tokens |
| batched target | 20.59 | 9.22 | 0.45× | 0.58× | 4 of 8 runs |

Lossless scalar results, averaged over the two passes:

| prompt | greedy tok/s | DFlash tok/s | ratio | emitted tokens/cycle |
|---|---:|---:|---:|---:|
| capital | 22.79 | 15.94 | 0.70× | 1.98 |
| fibonacci | 22.04 | 16.33 | 0.74× | 2.48 |
| math | 21.98 | 15.51 | 0.71× | 2.20 |
| story | 22.38 | 14.44 | 0.65× | 1.62 |

Batched bf16 diverged at output index **47 for capital** and **8 for story**
on both passes. Fibonacci and math matched all 100 tokens. Scalar verification
uses the same single-token target arithmetic as greedy decoding and stops
verification at the first rejected proposal; it removes these mismatches,
but also removes parallel target verification's amortization benefit.

Timings varied: capital baseline was 13.12 then 22.21 tok/s, and math DFlash
was 10.39 then 5.59 tok/s in the batch run. All receipts are retained without
discarding the slower trials. The host had existing swap usage; these are
measurements on a running desktop, not a controlled peak-throughput result.

An **FP32 diagnostic**, retaining the bf16 checkpoint values but changing
target arithmetic, matched all four 24-token smoke outputs: baseline 5.72,
DFlash 8.13 tok/s, mean paired ratio 1.42×, best 1.82×. It is slower in absolute
terms than the bf16 baseline and is not a bf16 speedup or a 100-token quality
validation. Its purpose is to investigate batch-dependent numerical drift.

## What was ported

The native C matrix backend is copied into `asahi/vendor/qwen3.c/`, with its
MIT license, pinned upstream revision and original file hashes. It builds
entirely here and has no runtime dependency on `~/qwen3.c` or `~/ane` source
files. The omarchy binary provenance checker is also copied with its license.
The installed MLX wheel and native Linux ANE driver are still prerequisites.

The local extension accepts resident **FP16 weights directly**, avoiding an
extra Q8 requantization of the trained draft. The context projector, Q/K/V/O
projections, gated MLP projections and vocabulary head execute on ANE. CPU
handles RMSNorm, RoPE, attention, SwiGLU, residuals and argmax. The target
remains on MLX/Vulkan. This is a draft offload, not full target offload.

The draft keeps only committed target features in its K/V cache. Proposal
K/V participate in the current block's attention and are then discarded.
Target verification returns every position's decision and features; rejected
cache positions are trimmed before the next cycle. The correction token is
the next unprocessed anchor. Sliding draft attention follows the checkpoint's
causal-within-block mask and anchor-relative context window.

A 32000-output vocabulary chunk passed the old backend's dimension checks
but **timed out on hardware**. The new FP16 API rejects N>8192 before issuing
the operation. The runner uses 38 chunks of at most 4096 outputs. This fixes
the workload here; it does not establish the hardware's maximum dimension.

Validation:

- Five FP16 matrix shapes, including K=1024/N=8192: CPU-product comparison
  passed, maximum absolute error 0.00129. Invalid widths and nonfinite inputs
  were rejected without a compute submission.
- Existing Q8 backend regression: 15 scalar cases and three batch cases,
  including in-place execution, passed; 27 submissions.
- Real target features, two incremental context updates: ANE versus CPU
  products with explicit FP16 input/output rounding gave hidden-vector cosine
  >0.9999998, relative L2 <0.00049, and **14/14 identical draft proposals**.
  This checks the projection backend on real inputs, not full equivalence to
  every framework's attention arithmetic.
- Cache tests cover full acceptance, rejection, mixed partial acceptance,
  a short final block and EOS inside an accepted block. They compare committed
  features with scalar decoding as well as token output.

## Why the macOS speed is still missing

The draft averages about **two emitted tokens per cycle** here. In the batch
run, target verification averaged 200.3 ms/cycle, draft body 23.9 ms, draft
head 23.6 ms, and context updates 3.7 ms. At the paired 20.59 tok/s baseline,
that cycle cost needs roughly **five tokens/cycle** to break even. This small
checkpoint's acceptance does not reach that threshold. Fibonacci accepted
2.48 tokens/cycle here versus the reported 7.6 in the original 4B experiment.
Those are different trained drafts and targets.

The copied kernel is useful, but it is only a fixed physical 32-row FP16
linear program. Short logical batches are zero-padded. It does not provide
the macOS stack's LUT6 compressed-weight execution, fused transformer graphs,
device-resident attention/cache operations, or target layer/head offload.
The native draft vocabulary head alone takes about 20–24 ms/cycle here;
the original palettized M4 Pro draft head was about 3 ms. Normalization and
attention also return to CPU between linear submissions.

The remaining work is concrete: make Vulkan batched bf16 target decisions
consistent with scalar decoding; reduce batched verification latency; use a
higher-acceptance draft trained for the target; and add native compressed
weights, transformer operations and efficient batch widths for further ANE
offload. Reproducing the **same** 4B bf16 stack also needs more memory than this
8 GB machine provides. Its target weights alone are about 8 GB, before the
draft, caches, runtime buffers and desktop. Driver availability and a working
linear kernel do not supply the missing runtime stack.

## Run it here

Use the existing `.venv-asahi` setup and pinned target cache described in
[asahi_findings.md](asahi_findings.md). Downloads, native build outputs and
draft weights stay under ignored `.asahi/` in this repository.

```bash
cd ~/mlx-ane-sd
bash asahi/build.sh
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY \
    scripts/asahi_python.sh scripts/fetch_asahi_dflash.py

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 \
    scripts/asahi_python.sh scripts/probe_asahi_ane.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 600 \
    scripts/asahi_python.sh scripts/check_asahi_dflash.py
timeout 600 scripts/asahi_python.sh scripts/test_asahi_sd_cache.py

# Lossless bf16 reference. It is expected to be slower than greedy decoding.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 900 \
    scripts/asahi_python.sh scripts/bench_asahi_dflash.py \
    --verify serial --max-new 100 --repeats 2 \
    --out notes/asahi/dflash_bf16_serial.json

# Batched research path; exit 2 means token identity failed.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 timeout 900 \
    scripts/asahi_python.sh scripts/bench_asahi_dflash.py \
    --verify batch --max-new 100 --repeats 2 \
    --out notes/asahi/dflash_bf16_batch.json
```

The harness acquires `~/ane.lock` then `~/gpu.lock` for the complete workload,
waiting for other cooperating workloads. Lock waits are outside decode timing.
The native register stream supports **base M1/T8103 only**; an accessible
device bound to the native `ane` driver is required. No root invocation or
silent CPU fallback is used.

Raw receipts: [scalar benchmark](asahi/dflash_bf16_serial.json),
[batch benchmark](asahi/dflash_bf16_batch.json),
[FP32 smoke](asahi/dflash_fp32_smoke.json),
[FP16 probe](asahi/ane_fp16_probe.json),
[real-input projection check](asahi/dflash_projection_check.json),
[Q8 regression](asahi/ane_q8_regression.log), and
[cache tests](asahi/sd_cache_tests.log).
