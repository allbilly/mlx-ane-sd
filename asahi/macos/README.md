# Matched M1 SD capture

The capture kit can be prepared after a Git pull on macOS, or prepared on Linux
and transferred, then executed on **this same base M1 under macOS**. It contains
the trained 0.6B DFlash weights needed for individual
graphs and real inputs from four benchmark prompts. It does not reproduce the
original M4 Pro / Qwen3-4B experiment.

The current prepared kit is `.asahi/m1-sd-capture-ready/`. All generated weights,
programs and captures stay under ignored `.asahi/` directories.

For the Git-pull workflow, follow [task.md](../../task.md). The four real input
fixtures are shipped in `asahi/fixtures/m1-sd/`; preparation downloads public
pinned weights and does not require a Linux ANE device on macOS. The task also
specifies the H13G command/register data needed for Linux replay.

## Prepare on Linux

```bash
cd ~/mlx-ane-sd
bash asahi/build.sh
timeout 180 scripts/asahi_python.sh scripts/profile_asahi_dflash.py
scripts/asahi_python.sh scripts/prepare_m1_sd_capture.py \
    --fixtures .asahi/macos-capture-inputs \
    --include-target --out .asahi/m1-sd-capture-new
python3 asahi/macos/check_manifest.py .asahi/m1-sd-capture-new
```

Use a fresh `--out` directory. `--include-target` copies the pinned 1.2 GB bf16
target for the matched Metal probe. The remaining kit needs only stdlib Python,
Xcode Command Line Tools and Apple's existing private frameworks on macOS.

Transfer the entire prepared directory or archive to the M1's macOS filesystem.
Retain all files, including model inputs and the manifest. After extraction,
enter that prepared directory; run **its** `tools/run_capture.sh`.

## Run on macOS

```bash
cd /path/to/m1-sd-capture-ready
python3 tools/check_manifest.py .
bash tools/run_capture.sh
```

The script verifies the base M1, builds the vendored Orion runtime locally,
checks input hashes, and runs the 20 cases serially in separate processes:

- RMSNorm, width 1024.
- Fused RMSNorm + gate/up projections + SwiGLU + down projection + residual,
  widths 1024/3072, physical sequence 32.
- Cached attention with 16 query heads, head dimension 128 and the actual
  committed context. Expanded GQA keys/values and RoPE are already present
  in the real input fixtures.
- Vocabulary-head chunks of 4096 and 8192 outputs, physical sequence 32.

Each is repeated for capital, Fibonacci, math and story inputs. This first kit
uses FP16 coefficients. LUT6/palette decoding, RoPE inside the graph, mutable
KV state, whole-draft graphs and target chunks require further captures and
runtime/compiler implementation.

The runner saves `cases/*/capture/` with:

- `output-0.bin`: output from the evaluated macOS ANE program.
- `evaluated-runtime/`: the complete temporary directory of that exact
  loaded program, copied before Orion removes it.
- `offline-hwx/`: a separate `ANECCompile` export targeting H13G, including
  its generated metadata and coefficient files.
- `receipt.json`: 20 warmed ANE evaluation timings and differences from the
  FP32 host diagnostic, plus export status.

**The offline HWX is a separate compilation.** Its decoded commands and
coefficients must be compared with the evaluated runtime artifacts before its
output is treated as the captured oracle. The host reference is a diagnostic,
not an Apple golden or a full SD quality gate. A failed case leaves its log and
partial artifacts; the shell script returns nonzero.

`capture-manifest.json` records artifact hashes. `build/` records macOS version,
chip and compiler. Private frameworks are used from the installed OS; they
are not copied into the kit.

## Measure the same target on Metal

This is the missing matched GPU comparison. Use stock MLX in a separate venv:

```bash
python3 -m venv .venv-metal
.venv-metal/bin/python -m pip install 'mlx==0.32.2' 'mlx-lm==0.31.3' numpy
.venv-metal/bin/python tools/metal_reference.py --repeats 2
```

The script verifies the transferred target weights and metadata, benchmarks
four prompts at 100 new tokens, and measures query widths 1/2/4/8/16/32 on
fixed teacher-forced histories. It includes the same synchronous cache
evaluation and feature transfer as the Linux runner. Output is
`metal-reference.json`; mean/max rates and token identity are explicit.
Use `--model` for an independently transferred copy of the identical snapshot.

## Bring results back to Linux

Transfer the complete directory back, including `capture-manifest.json`,
`metal-reference.json`, inputs and all runtime/HWX files. Check the original
input hashes again with `check_manifest.py`.

Next work is to decode H13G tasks, bindings, strides, scratch surfaces and
coefficients, compare both compiler exports, then replay with the Linux driver
against `output-0.bin`. Hardware replay of these new cases and macOS execution
of this kit are still pending. Compilation or a finite host-reference result
alone does not establish replay correctness or SD speedup.
