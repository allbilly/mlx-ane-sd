# Compact M1 full-ANE reference

This kit preserves the kernels for the measured M1 macOS result: **72.00 tok/s
SD versus 45.19 tok/s MLX BF16 Metal (1.593×)** across four prompts, two passes
and 100 generated tokens. The full-stack runner uses this kit and the public
checkpoints. Older per-operation capture bundles are optional local debugging
material and are kept outside Git.

The full-stack reference is about **5.54 MB**. It contains no learned weight
banks, embedding, compiler `weight.bin`, complete weight-bearing HWX files, or
captured input/output binaries. Public model checkpoints are downloaded
separately. Large original captures remain local and are unnecessary for Linux
reconstruction.

## What is retained

- Five compressed kernel templates with learned coefficients zeroed: two
  14-layer target chunks, the three-layer DFlash draft, context projector and
  complete shared 151,936-token vocabulary head.
- Compact quantizer boundaries and packing schedules. Cluster indices and
  palette means are rebuilt from BF16 safetensors; rare palette rounding
  corrections retain the exact original LUT6 choices.
- Original MIL graph source, compiler tensor layouts and Linux BAR/buffer plans.
  Templates retain the full command stream, symbols and fixed activation tables.
  The decoder can regenerate the 6,034 tasks and 763,960 register writes.
- Hashes, shapes and traces for 72 real calls and 588 output tensors across all
  four prompts, covering early, decode and populated-cache calls. The host loop
  regenerates their inputs and verifies outputs by hash.
- A 40 KB fixed mathematical RoPE table. This prevents NumPy version differences
  in sine rounding from changing an FP16 input. It contains no learned weights.
- Pinned public checkpoint revisions, checkpoint hashes, full reconstructed HWX
  hashes and the original native benchmark identity.

All five full HWX files reconstruct **byte-for-byte** from these templates and
the external checkpoints. Regenerated inputs and all 588 outputs also match
the captured CoreML ANE goldens. See
[reconstruction.json](reconstruction.json) and
[macOS validation](../../notes/m1_compact_reference_validation.json).

## Inspect and reconstruct

From the repository root, select the existing Python/NumPy environment for
the current OS. Planning and reconstruction use only NumPy and the standard
library. On this macOS workspace the environment is `.asahi/venv-metal/`;
on Asahi use `.venv-asahi/`:

```bash
# macOS:
replay_python=.asahi/venv-metal/bin/python
# On Asahi Linux, use this assignment instead:
# replay_python=.venv-asahi/bin/python

"$replay_python" scripts/package_m1_full_stack.py verify artifacts/m1-full-ane
"$replay_python" scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode plan

hf download mlx-community/Qwen3-0.6B-bf16 \
  --revision 42096995f6402fde107068cf530136fe64b604f8 \
  --local-dir .asahi/models/m1-full-ane-target \
  --include 'config.json' 'generation_config.json' 'tokenizer*' \
            'special_tokens_map.json' 'model.safetensors'
hf download orestis-z/dflash-qwen3-0.6b-microcycle-dflash \
  --revision 4dd1e04078f993593338ef1f9403179e41e4580e \
  --local-dir .asahi/models/m1-full-ane-draft \
  --include 'config.json' 'model.safetensors'

# Works offline on macOS or Linux once the checkpoints have been downloaded.
"$replay_python" scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode reconstruct \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft
```

`plan` opens no ANE device. `reconstruct` requires no CoreML, Torch or Metal;
it checks both checkpoint hashes and all five complete HWX hashes, then caches
the approximately 502 MB runtime binaries in ignored
`.asahi/m1-full-ane-cache/`. These are runtime model assets, not Git material.
Allow that cache space in addition to the two public checkpoints. The existing
extract command can copy the small kit elsewhere; extraction of bulk captures
is unnecessary.

The exact-reference loader supports the pinned **BF16 safetensors** models.
A quantized GGUF checkpoint would change their values and cannot claim these
exact reference hashes.

## Verify on Asahi, then benchmark

Boot base M1 Asahi with an accessible `ane` accel device implementing the
[allbilly/ane submit ABI](https://github.com/allbilly/ane/blob/main/kmod/uapi/drm/ane_accel.h).
The runner checks t8103 and the driver and waits for exclusive access through
the existing `~/ane.lock` and `~/gpu.lock` reservations.

```bash
scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode verify \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft \
  --out notes/m1_linux_full_ane_verify.json

scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode bench \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft \
  --transport resident --head-readback native \
  --verify-readback accepted-prefix --kv-readback prefix \
  --native-threads 1 --cpu-affinity 4,5,6,7 --cpu-util-min 1024 \
  --warmup-new 100 --max-new 100 --repeats 4 \
  --out notes/m1_linux_full_ane_bench.json
```

The optimized command requires a C compiler with OpenMP. Its CPU utilization
hint applies to the benchmark process and its new workers, equally for all
three configurations; it changes no system settings. Each run requires a fresh receipt path. Verification generates all four
100-token SD traces, checks every selected input/output hash and refuses a
benchmark after a mismatch. The default Linux transport keeps target/draft K/V
buffers resident and writes only committed cache positions. It reduces useful
vocabulary rows directly from mapped output, while verification checks complete
physical outputs and compares reductions with full readback. Select
`--transport reference --head-readback full` for the original full-cache packing
and full-output readback control; receipts record both options.
The benchmark repeats verification, compares Linux
stock greedy MLX BF16 (`mlx_lm.stream_generate`, temperature zero,
`prefill_step_size=32`) with LUT6 ANE AR and SD, and enforces SD/AR token
identity. This is the same baseline and reported `generation_tps` metric used
in the macOS receipt. Before measured trials, all three configurations run each
of the four prompts for up to 100 tokens; those warm-up calls are recorded
separately and excluded from benchmark rows and speedup calculations. Partial
results and failures are saved, including failures during warm-up.

Receipts document timing per configuration: the stock MLX rate excludes time
to the first token, while ANE's host decode timer includes the last prompt-token
forward. These follow the original macOS benchmark conventions. Loading,
prefill and warm-up are excluded from reported throughput.

## Scope and remaining hardware work

The retained kernel templates come from **offline HWX compiler exports**.
Direct macOS `_ANEClient` loading rejected those exports at stage 4 with status
`0x1`. The corresponding MIL graphs executed on ANE and matched CoreML goldens;
this does not establish execution of the exported HWX itself on macOS. Asahi
now executes all five exports and matches all 588 selected output hashes and
four complete SD traces. After native useful-row readback and process scheduling
changes, its four-prompt, four-pass confirmation averages **61.39 tok/s SD
versus 36.22 tok/s MLX BF16 (1.695×)**. All 1,600 measured SD tokens match
macOS. The initial Linux result was 0.776×; macOS measured 1.593×.
See [the Linux report](../../notes/m1_asahi_full_ane_results.md) and
[the confirmed recipe and receipts](../../notes/m1_asahi_scheduler_results.md).

The target is LUT6 compressed and differs from BF16 on near-tie logits. SD
matches the same compressed target. Its ANE AR control uses padded B=8 kernels;
an optimized B=1 ANE baseline remains unmeasured. Linux uses a Python host loop
and an MLX Vulkan baseline, while the 72 tok/s macOS result used Swift and
Metal. These measurements do not reproduce the original M4 Pro / 4B result. See the
[handoff](../../notes/m1_full_ane_asahi_handoff.md).
