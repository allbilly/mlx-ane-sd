# M1 Asahi support

The Linux runner executes the complete Qwen3-0.6B DFlash stack on the M1 ANE:
two 14-layer target chunks, the draft context projector, three-layer draft and
shared full-vocabulary head. The Python host supplies embeddings, cache updates
and acceptance decisions. MLX/Vulkan runs the stock BF16 GPU baseline.

The confirmed measurements and per-prompt comparison extend the
[main README table](../README.md#current-best-macos-and-asahi). They use four
prompts, four passes and 100 generated tokens per trial. The public model pair
fits this 8 GB M1; these measurements are separate from the M4 Pro/Qwen3-4B
experiment.

## Environment

The tested machine is a base M1 MacBook Air (T8103), 8 GB, running Fedora Asahi
Remix 44 and kernel `7.1.13+`. The [ANE driver](https://github.com/allbilly/ane)
must be loaded and `/dev/accel/accel0` accessible to your account. The runner
checks the chip and submit ABI. Native readback also needs a C compiler with
OpenMP and FP16 NEON support.

Use an isolated Python 3.14 environment for the pinned Vulkan MLX wheel.
Install `mlx-lm` without dependencies so it cannot install upstream Metal MLX
beside the Vulkan fork. Run these commands from the repository root:

```bash
python3.14 -m venv .venv-asahi
uv pip install --python .venv-asahi/bin/python --no-cache \
  'https://github.com/joshuaswarren/omarchy-mlx/releases/download/v0.7.27/mlx_omarchy-0.32.4.dev202610041653%2B6edd258-cp314-cp314-linux_aarch64.whl#sha256=f10c1df8677d1ada2a438c7ec0d40b50fc3ca75b5cfdb2be8a697f9482f09b87'
uv pip install --python .venv-asahi/bin/python --no-deps mlx-lm==0.31.3
uv pip install --python .venv-asahi/bin/python \
  numpy==2.5.3 transformers==4.57.6 huggingface-hub==0.36.2 \
  safetensors protobuf pyyaml jinja2 tqdm
```

The wheel needs `libopenblas.so.0` and `libgfortran.so.5`. They may come from
system packages or from extracted Fedora RPMs under
`.venv-asahi/native/usr/lib64`. [The original setup receipt](../notes/asahi/setup.json)
records package identities and hashes. `scripts/asahi_python.sh` adds that
optional library directory when starting Python. See the
[MLX provenance receipt](../notes/asahi/m1_full_ane_mlx_provenance_20261006.json)
and [preflight](../notes/asahi/m1_full_ane_preflight_20261006.json).

## Download the pinned public weights

Use `hf download`; no compiled macOS models or bulk tensor dumps are needed:

```bash
source .venv-asahi/bin/activate
hf download mlx-community/Qwen3-0.6B-bf16 \
  --revision 42096995f6402fde107068cf530136fe64b604f8 \
  --local-dir .asahi/models/m1-full-ane-target \
  --include 'config.json' 'generation_config.json' 'tokenizer*' \
            'special_tokens_map.json' 'model.safetensors'
hf download orestis-z/dflash-qwen3-0.6b-microcycle-dflash \
  --revision 4dd1e04078f993593338ef1f9403179e41e4580e \
  --local-dir .asahi/models/m1-full-ane-draft \
  --include 'config.json' 'model.safetensors'
```

The runner checks checkpoint hashes before decoding coefficients. The two BF16
safetensors total about 1.92 GB; reconstruction adds about 502 MB of runtime
HWX. The [5.54 MB reference kit](../artifacts/m1-full-ane/README.md) stores
weight-free templates and packing recipes. Quantized GGUF values would change
the measured weights and cannot reproduce these exact hashes.

## Inspect, reconstruct and verify

`plan` opens no device, and `reconstruct` uses only NumPy and the standard
library. Both also work on macOS in a Python/NumPy environment. Planning does
not require downloaded models or a `.asahi/` directory.

```bash
scripts/asahi_python.sh scripts/package_m1_full_stack.py verify artifacts/m1-full-ane
scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode plan
scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode reconstruct \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft
scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode verify \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft \
  --out notes/m1_linux_full_ane_verify_new.json
```

Verification regenerates all four full SD traces and checks 72 selected calls /
588 output hashes against macOS, including physical input padding. The five
complete HWX hashes must also match. A mismatch stops execution and saves a
failure receipt. Compiled CoreML models, Torch and Metal are unnecessary for
Linux reconstruction and execution.

## Reproduce the confirmed benchmark

```bash
scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode bench \
  --target .asahi/models/m1-full-ane-target \
  --draft .asahi/models/m1-full-ane-draft \
  --transport resident --head-readback native \
  --verify-readback accepted-prefix --kv-readback prefix \
  --native-threads 1 --cpu-affinity 4,5,6,7 --cpu-util-min 1024 \
  --warmup-new 100 --max-new 100 --repeats 4 \
  --out notes/m1_linux_full_ane_bench_new.json
```

Use fresh receipt paths. Verification runs again before timing. The runner
waits for exclusive `~/ane.lock` and `~/gpu.lock` reservations, warms all three
configurations for every prompt, reverses their order between passes and
synchronizes MLX outside the generation timer. Loading, prefill and warmups
are excluded. The MLX metric excludes time to the first token; ANE decode
includes the final prompt-token forward, following the original macOS receipt.

Resident caches upload only committed rows. Native readback reduces FP16
vocabulary outputs directly from mapped buffers and reads target predictions
through the first rejected candidate or bonus token. Affinity and the process
utilization hint apply equally to MLX, ANE AR and ANE SD, including new workers.
They change no system governor or other process. `--transport reference
--head-readback full` selects the earlier full-cache/readback control.

The [four-pass confirmation receipt](../notes/m1_linux_full_ane_bench_uclamp_confirm_warm100_20261006.json)
contains all 48 measured trials. All 1,600 SD tokens and acceptance/cache
choices match the macOS compressed-target reference. Mean paired speedup is
1.696×, maximum paired speedup 1.954× and maximum SD trial 69.12 tok/s. The
[profile receipt](../notes/asahi/m1_full_ane_profile_uclamp_prefix_20261006.json)
records phase costs separately from benchmark throughput.

The target is LUT6 compressed, so near-tie logits may differ from BF16.
SD must match the same LUT6 ANE AR target. That AR control uses padded B=8
programs; a separately compiled B=1 ANE baseline remains unmeasured.

## Optional M1 macOS reference benchmark

The macOS conversion and Linux adapters share the Python host loop and common
BF16/tensor helpers. The native Swift runner supplies the historical macOS
comparison. For a new macOS run, use Python 3.11 and Apple's compiler:

```bash
python3.11 -m venv .venv-macos
.venv-macos/bin/python -m pip install \
  numpy==1.26.4 torch==2.5.1 coremltools==9.0 \
  mlx==0.32.2 mlx-lm==0.31.3 huggingface-hub==1.33.0 \
  transformers==5.18.0 safetensors==0.8.0 tokenizers==0.23.2
source .venv-macos/bin/activate
# Use the same pinned hf download commands above, then:
.venv-macos/bin/python scripts/convert_macos_m4_recipe.py --out .asahi/m4-recipe-m1-new
swiftc -O -framework CoreML swift-bench/m1_full_ane.swift \
  -o .asahi/m4-recipe-m1-new/m1-full-ane
.venv-macos/bin/python scripts/bench_macos_m4_recipe.py --native \
  --artifacts .asahi/m4-recipe-m1-new --max-new 100 --repeats 2 \
  --out notes/m1_m4_recipe_new.json
```

Conversion uses the current environment directly. It checks cache provenance
and ANE math placement; fresh artifacts can differ after recompilation or
palettization. The [original macOS receipt](../notes/m1_m4_recipe.json),
[completed historical audit](../notes/m1_m4_recipe_audit.json) and
[compact-reference validation](../notes/m1_compact_reference_validation.json)
remain unchanged. They describe the measured historical code, rather than a
new hardware run of this refactor.

The complete research history, earlier unsuccessful routes, source snapshots,
capture tools and cleanup receipts remain on the
[asahi research branch](https://github.com/allbilly/mlx-ane-sd/tree/asahi).
The reference source state is commit `f17a674`. The ignored `.asahi/` cache may
be removed in full after use; subsequent runs download weights and reconstruct
kernels again. No large captured weights or tensor dumps need to be transferred.
