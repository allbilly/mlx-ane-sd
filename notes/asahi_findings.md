# Asahi Linux compatibility and SD test

**Date:** 2026-10-05. **Hardware:** Apple MacBook Air M1, 8 GB.
**OS:** Fedora Asahi Remix 44, kernel 7.1.13+.

MLX GPU execution works through [omarchy-mlx](https://github.com/joshuaswarren/omarchy-mlx).
The small-model SD benchmark is slower than greedy decoding and fails strict
token identity on three of four prompts. The original macOS full-ANE DFlash
stack was not reproduced.

Follow-up: a trained small DFlash draft now executes on the native M1 ANE
using a backend copied into this repository. See
[asahi_dflash_findings.md](asahi_dflash_findings.md) for the lossless reference,
batched numerical failures, measured slowdown and build/run instructions.

## Measured results

Qwen3-0.6B **bf16 target**, Qwen3-0.6B **4-bit autoregressive draft**,
`num_draft_tokens=3`, `max_new=100`, greedy sampling, raw prompts. These
are ordinary mlx-lm speculation tests, not the trained DFlash architecture.
The target is unquantized bf16; this does not test a 4-bit target.

Each prompt ran twice with baseline/SD order reversed in the second pass.
Both paths warmed for eight tokens. Each run used fresh caches. Prefill of
all but the final prompt token was outside decode timing for both models;
every decode block, including the first, was timed. All measured runs
generated 100 tokens. The desktop remained active, with no governor changes
or forced cooling. Available governor and memory snapshots are in the receipt;
no readable thermal-zone temperatures were available.

| prompt | baseline tok/s | SD tok/s | token IDs identical | first difference (0-based) |
|:--|--:|--:|:--|--:|
| capital | 33.31 | 10.98 | no | 3 |
| fibonacci | 33.21 | 10.88 | yes | — |
| math | 34.86 | 11.14 | no | 29 |
| story | 33.97 | 10.28 | no | 96 |
| **mean** | **33.83** | **10.82** | **1/4 prompts** | |

Mean paired speedup: **0.320×**; maximum: **0.333×**. Draft-origin token
fractions were 72%, 73%, 72%, and 62%, respectively, in both passes. Both
passes diverged at the same positions. The harness completed and exited
**2** because token equality failed; the failure was not waived.

The model, hardware, backend, draft architecture, and timing procedure differ
from the M4 Pro/macOS benchmarks. These numbers are not a cross-OS speedup
comparison or evidence about the DFlash speedup ceiling.

## Functional and numerical checks

The loaded MLX core extension and `libmlx.so` matched the wheel RECORD hashes.
The wheel itself matched the release SHA256SUMS. Device identity was
`Device(gpu, 0)`, Apple M1 (G13G B1), Honeykrisp, Mesa 26.2.3, Vulkan 1.4.354.
The stock driver reports a 3,932,160,000-byte usable memory heap. No device
override or simulated capabilities were enabled. Mesa supplied no git SHA.

GPU smoke checks passed for fp32, fp16, and bf16 matmul, RMSNorm, RoPE, and
causal SDPA. The small matmul fixture's maximum absolute errors were 0, 0,
and 0.015625. A separate **untrained** dense-Qwen3 shape fixture tested the
acceptance and rejection/cache paths: an identical draft accepted 12/16
tokens; a different draft accepted 0/16; both reproduced greedy target IDs.
This fixture establishes execution compatibility, not pretrained-model
quality. Vulkan dispatch traces accompany both the fixture and a short
pretrained baseline/SD run. The driver buffer round trip also passed.

The pretrained target loaded with bf16 weights and generated
“Paris. The capital of Italy is Rome” from “The capital of France is”.

The numerical probe uses **real benchmark prefixes**, with identical cached
histories and the same bf16 target. It compares final input tokens evaluated
one at a time versus in a block of up to four. It contains no draft model,
quantized weights, or speculative acceptance loop. It records actual argmax
IDs, top-five logits, logit gaps, and full-vector errors. This isolates changes
in target evaluation from draft quantization. See the probe receipt for which
argmax changes it reproduces; a controlled final block does not replay every
earlier speculative cache update.

The controlled probe reproduced capital (`such` → `like`, single-token top
gap 0.125, block tie) and math (`2` → `4`, single-token tie, block gap 0.125).
Full-vector maximum logit differences were 0.158 and 0.125. Fibonacci kept
its winner with a 12.875 logit gap. Story's final-block probe retained its
winner despite a maximum logit difference of 0.25; its full-run divergence
was not isolated by this probe. These observations support a numerical
near-tie explanation for capital and math; they do not establish bit-exact
target verification or fully explain story.

## ANE and full-stack boundary

The Linux ANE driver is bound (`apple,t8103-ane`) and `/dev/accel/accel0` is
present. This original MLX-only experiment established device availability
without running a DFlash model on ANE. The installed CoreML CLI exposes
package `inspect` and `check`; the macOS runner uses Apple's Swift CoreML
framework and `.mlmodelc` artifacts. Those artifacts still need a Linux
execution adapter. The later native DFlash port bypasses CoreML using the
copied C matrix backend, as documented in the follow-up report above.

The original Qwen3-4B bf16 target is about 8 GB before its approximately 1 GB
draft and caches. That configuration is unsuitable for a useful benchmark on
this 8 GB machine, so a smaller dense Qwen3 target was used. The existing
macOS runners and their results were left intact.

## Environment and reproduction

The isolated environment is `.venv-asahi`, Python 3.14.7. Runtime:
`mlx-omarchy==0.32.4.dev202610041653+6edd258`, release **v0.7.27**;
`mlx-lm==0.31.3`, `transformers==4.57.6`. The omarchy checkout at
`../omarchy-mlx` supplies its provenance checker; override its location with
`--omarchy-repo` if needed.

Never install upstream `mlx` alongside the Vulkan wheel. Install mlx-lm with
`--no-deps`, then install its support packages separately:

```bash
python3.14 -m venv .venv-asahi
uv pip install --python .venv-asahi/bin/python --no-cache \
  'https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.27/mlx_omarchy-0.32.4.dev202610041653%2B6edd258-cp314-cp314-linux_aarch64.whl#sha256=f10c1df8677d1ada2a438c7ec0d40b50fc3ca75b5cfdb2be8a697f9482f09b87'
uv pip install --python .venv-asahi/bin/python --no-deps mlx-lm==0.31.3
uv pip install --python .venv-asahi/bin/python \
  'numpy==2.5.3' 'transformers==4.57.6' 'huggingface-hub==0.36.2' \
  safetensors protobuf pyyaml jinja2 tqdm
```

The wheel needs `libopenblas.so.0` and `libgfortran.so.5`. This session
extracted Fedora `openblas-serial` and `libgfortran` RPMs into
`.venv-asahi/native/`, without changing system packages. The wrapper adds
`.venv-asahi/native/usr/lib64` to its library search path. RPM identities and
library hashes are saved in `asahi/setup.json`.

The configured HTTP proxy repeatedly timed out on downloads. Direct
connections worked. The weight downloads used a ModelScope mirror and were
verified against the pinned Hugging Face LFS hashes before becoming available
to the loader. A normal Hugging Face download can populate the same cache:

```bash
env -u HTTPS_PROXY -u HTTP_PROXY HF_HUB_DISABLE_XET=1 scripts/asahi_python.sh - <<'PY'
from huggingface_hub import snapshot_download
for model, revision in [
    ('mlx-community/Qwen3-0.6B-bf16', '42096995f6402fde107068cf530136fe64b604f8'),
    ('mlx-community/Qwen3-0.6B-4bit', '73e3e38d981303bc594367cd910ea6eb48349da8'),
]:
    snapshot_download(model, revision=revision,
                      allow_patterns=['*.json', '*.safetensors', '*.txt'])
PY

timeout 90 scripts/asahi_python.sh scripts/smoke_asahi_mlx.py
flock -w 30 /tmp/m1-gpu.lock timeout 600 \
  scripts/asahi_python.sh scripts/bench_asahi_mlx.py \
  --max-new 100 --num-draft 3 --repeats 2 --out notes/asahi/bench.json
timeout 180 scripts/asahi_python.sh scripts/probe_asahi_verify.py
```

The benchmark loads only cached revisions, checks binary provenance, verifies
identical tokenizer hashes and a dense bf16 target, and checkpoints results
after each prompt. It exits 2 on token divergence. The session also acquired
the host's existing `gpu.lock` to coordinate GPU use.

## Receipts

- [bench.json](asahi/bench.json): exact command, harness hash, source commit,
  binary/model hashes, runtime versions, driver identity, timings and all tokens.
- [bench.log](asahi/bench.log): both passes and the failed identity result.
- [hardware.json](asahi/hardware.json), [setup.json](asahi/setup.json),
  [mlx_provenance.json](asahi/mlx_provenance.json): device and installation identity.
- [smoke.log](asahi/smoke.log): GPU fixture checks and Vulkan dispatch trace.
- [driver_smoke.log](asahi/driver_smoke.log): device buffer round trip.
- [driver_reopen.log](asahi/driver_reopen.log): successful fresh-process device
  reopen and buffer round trip after all inference tests.
- [model_load_smoke.log](asahi/model_load_smoke.log): pretrained generation check.
- [verify_probe.json](asahi/verify_probe.json): single-token vs block target logits.
- [trace_bench.json](asahi/trace_bench.json), [trace_bench.log](asahi/trace_bench.log):
  short traced pretrained run. Its rates include tracing overhead and are
  diagnostic only; they are excluded from the reported benchmark numbers.
