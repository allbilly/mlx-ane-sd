# Task: matched M1 macOS measurements and ANE dumps for Asahi SD

Use the **same base M1 MacBook Air (T8103, 8 GB)** booted into macOS to obtain
the missing Metal comparison and real ANE programs for Linux replay. Work only
in this repository. After pulling the committed changes, follow the steps
below; the ignored `.asahi/` directory from Linux is not required.

Include the supporting `asahi/` files, new Python helpers in `scripts/`, and
referenced `notes/` receipts in the same commit as this task. In particular,
the four fixtures, vendored Orion runtime, capture tools, preparer/downloader
and `notes/asahi/dflash_bf16_serial.json` are required after Git pull.

The immediate deliverable is a reproducible capture bundle, followed by an
Asahi implementation and correctness/performance measurements. macOS execution
and replay of these new cases have not yet been tested. Keep failures and logs
as evidence rather than marking compilation as a successful replay.

## Why this task is needed

The original 2.21× result used an M4 Pro with 64 GB, Qwen3-4B bf16, a different
DFlash draft, and fused CoreML/LUT6 graphs. Our M1 experiment uses Qwen3-0.6B
bf16 and a trained three-layer, eight-token draft. The Linux draft currently
makes **59 ANE submissions per proposal**, with attention, norms and activations
on CPU. These are materially different execution paths.

On four prompts, two passes and 100 new tokens, Linux scalar verification
preserves all 800 greedy tokens but slows from **22.30 to 15.56 tok/s**. Batched
verification gives **20.59 to 9.22 tok/s**, with only four of eight runs token
identical. The fixed-history width-eight target call takes about 164 ms on
stock Vulkan. The native draft profile averages 39.85 ms/proposal, including
17.23 ms inside synchronous submission calls; that span includes compute and
waiting, so it is not pure driver overhead.

See [the investigation](notes/asahi_sd_project_survey.md) and
[DFlash findings](notes/asahi_dflash_findings.md). Measure the GPU gap and the
ANE graph gap separately before combining changes into the SD loop.

## 1. Prepare after Git pull on macOS

Prerequisites: native arm64 Python **3.11 or newer**, Xcode Command Line Tools,
network access for public model downloads, and several GB of free disk space.
The capture script checks that the chip is exactly `Apple M1`. Run hardware
benchmarks serially, on AC power, with other inference workloads stopped.

```bash
cd ~/mlx-ane-sd
git pull --ff-only
uname -m
sysctl -n machdep.cpu.brand_string
xcrun clang --version
python3 -c 'import sys; assert sys.version_info >= (3, 11), sys.version'

mkdir -p .asahi
python3 -m venv .asahi/venv-metal
.asahi/venv-metal/bin/python -m pip install \
    'mlx==0.32.2' 'mlx-lm==0.31.3' numpy huggingface-hub

# Fetch the pinned draft; the downloader checks all four file hashes.
.asahi/venv-metal/bin/python scripts/fetch_asahi_dflash.py

# Populate the pinned target snapshot, using the recorded file list.
.asahi/venv-metal/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import snapshot_download
target = json.loads(Path('notes/asahi/dflash_bf16_serial.json').read_text())['target']
print(snapshot_download(target['model'], revision=target['revision'],
                        allow_patterns=list(target['files'])))
PY

# Prepare a fresh kit. Its default fixtures are now shipped in Git.
.asahi/venv-metal/bin/python scripts/prepare_m1_sd_capture.py \
    --include-target --out .asahi/m1-sd-macos
.asahi/venv-metal/bin/python asahi/macos/check_manifest.py .asahi/m1-sd-macos

mkdir -p .asahi/m1-sd-macos/build
git rev-parse HEAD > .asahi/m1-sd-macos/build/repo-commit.txt
git status --short > .asahi/m1-sd-macos/build/repo-status.txt
.asahi/venv-metal/bin/python -m pip freeze \
    > .asahi/m1-sd-macos/build/python-packages.txt
```

Use a new `--out` name for a repeat; preparation refuses to overwrite a nonempty
kit. Keep an existing kit intact if it contains results. Do not build the Linux
C/DRM backend on macOS. The preparer reads weights and recorded tensors without
opening a Linux ANE device.

Pinned inputs:

| Component | Model / source | Revision |
|---|---|---|
| bf16 target | `mlx-community/Qwen3-0.6B-bf16` | `42096995f6402fde107068cf530136fe64b604f8` |
| DFlash draft | `orestis-z/dflash-qwen3-0.6b-microcycle-dflash` | `4dd1e04078f993593338ef1f9403179e41e4580e` |
| macOS runtime | vendored `allbilly/Orion` | `6aee791010f554232140eef3d730bb8410d090e7` |
| Real inputs | `asahi/fixtures/m1-sd/{capital,fibonacci,math,story}.npz` | Per-file SHA256 in `SOURCE.json` |

Target file hashes are in the Linux benchmark receipt and generated manifest;
draft hashes are in `scripts/fetch_asahi_dflash.py`. Preserve the target bf16
weights, tokenizer, prompts and feature IDs **1, 13, 25** (after decoder layers
0, 12, 24). Model downloads and generated programs stay under ignored paths.

## 2. Measure the matched Metal target first

```bash
.asahi/venv-metal/bin/python .asahi/m1-sd-macos/tools/metal_reference.py \
    --repeats 2 > .asahi/m1-sd-macos/metal.log 2>&1
```

Required result: `metal-reference.json` with four prompts × two passes, 100 new
tokens, mean/max decode rates, token IDs and identity against the Linux greedy
traces. The same script measures widths **1, 2, 4, 8, 16, 32** on fixed teacher
histories, including target hidden-feature transfer. It reports block-vs-scalar
identity within macOS too. Record mismatches; do not hide them by substituting
a different precision or target.

Compare these numbers with `notes/asahi/sd_routes.json`. This establishes whether
small-block verification is faster on Metal on the same chip. It does not yet
measure a macOS DFlash SD loop.

## 3. Execute and capture the 20 ANE cases

```bash
bash .asahi/m1-sd-macos/tools/run_capture.sh
```

The script builds the vendored runtime and runs five graphs for each of the
four real prompt fixtures. Each successful graph receives two warmup evaluations
and 20 timed evaluations.

| Graph to capture | Shape / purpose |
|---|---|
| RMSNorm | Hidden width 1024, physical sequence 32; actual eight draft rows |
| Fused FFN | RMSNorm → gate/up → SwiGLU → down → residual; hidden 1024, intermediate 3072, sequence 32 |
| Cached attention | 16 query heads, head dimension 128, eight real queries padded to 32; real context bucket padded to a multiple of 32 |
| Head chunk | 1024 → 4096 vocabulary logits, seven real rows padded to 32 |
| Wider head chunk | 1024 → 8192 vocabulary logits, same inputs; tests whether fewer head submissions are viable |

Attention inputs already contain Q/K normalization, RoPE and expanded GQA K/V.
These first graphs use FP16 coefficients. They establish actual shape support,
fusion, layouts and replay goldens before adding a whole draft or LUT6.

For every case retain `model.mil`, `case.json`, input tensors, coefficient BLOBs,
host diagnostic, `macos.log`, and all files under `capture/`:

- `output-0.bin`: output from the program actually evaluated on macOS ANE.
- `evaluated-runtime/`: the exact model's temporary directory, copied before
  unload/cleanup. Inspect which executable files Apple actually placed there.
- `offline-hwx/`: a **separately compiled** H13G `model.hwx`, compiler metadata,
  status dictionary and any additional coefficient files.
- `receipt.json`: warmed ANE latency, finite-output checks, error against the
  host diagnostic, and the independent HWX export status.

The shell script continues across failed cases, retains partial artifacts and
returns nonzero when any case fails or its HWX export is incomplete. Report
those cases explicitly. An offline export is not an evaluated oracle: compare
its decoded program and coefficients with the evaluated runtime artifact.
If the runtime directory contains only source/cache metadata, record that the
loaded executable remains missing and perform the tracing step below.

## 4. Exact ANE dump and kernel/register data needed

The essential dump is **a complete H13G program plus its data/bindings and a
golden output for the same input**. A `.mlmodelc` directory, register screenshot,
or HWX without coefficients and bindings is insufficient for replay.

### Compiler program and task descriptors — required for each graph

Decode both the evaluated-runtime executable, when available, and the offline
HWX. Save `container.json`, raw `task-descriptors.bin`, `tasks.json` and
`registers.json` alongside the original binaries. This decoding is follow-up
work; the current capture runner retains files but does not produce these
decoded reports or read live MMIO registers.

Retain:

1. Container architecture/ISA, load commands, entry point, all segment/section
   sizes and offsets, thread/BAR table, symbols and relocation information.
   Expect M1/H13G, CPU subtype 4 and v7 task encoding; reject M4/H16 programs.
2. Every task's complete raw header and ordered command packets, task ID,
   network/end flags, `NextPointer`/`NextSize`, entry descriptor size, task
   count, dependency/events/exception fields and all active base selectors
   (`RBase`, `WBase`, `TBase`, `KBase`). Preserve unknown words and padding.
   Tasks can vary in size; do not impose the current linear template's size.
3. Every register write as **task ID + packet order + engine-relative byte
   offset + raw value**, with decoded names/fields where known. A final map
   alone can lose repeated writes. Keep the raw stream as the authority.

These M1 command-register blocks are the priority:

| Block / M1 engine-relative base | Fields needed and reason |
|---|---|
| Common `0x00000` | Input/output dimensions and formats, `Cin/Cout`, convolution/group/tile configuration, active engines, task/context flags; explains tiling and useful work |
| L2 `0x04800` | Source/result modes, SRAM bases, channel/row strides, aliases and buffer modes; enables intermediate reuse across fused tasks |
| Planar Engine `0x08800` | Operation/condition/source selection, activation, bias and scales; required for norm, SwiGLU and residual scheduling |
| Neural Engine `0x0C800` | Kernel format, MAC mode, nonlinear mode, accumulation/bias/scales; later LUT/palette enable and bit width |
| TileDMA source `0x13800` | Base offset, row/plane/depth/group strides, format/interleave and dependencies; reconstructs exact input layout |
| TileDMA destination `0x17800` | Destination base/strides/format, buffer mode, padding and end-of-write flags; reconstructs output layout and completion |
| Coefficient DMA `0x1F800` | All enabled coefficient slots, bases, lengths, format/cache/prefetch controls; identifies coefficient packing and residency |

These are command programming offsets, not physical SoC addresses. Decode the
H13 stream packet header separately from the register values. Preserve complete
blocks, including fields whose meaning has not been established. The local
survey identified `coreml_to_ane_hwx/hwx_dump/h13_register_map.md` and
`ane/gpt2/hwx.py` as references; copy any needed implementation into this repo
with license/provenance rather than changing or requiring those other repos.

### Buffer bindings and payloads — required for replay

Save `bindings.json` and all associated binary payloads. For every program,
coefficient, constant, input, output and scratch buffer record its role, logical
BAR/bank selector, segment offset, byte size, alignment, dtype, shape, physical
strides and SHA256. Record aliases, initial scratch/state contents and required
zero padding. Associate macOS input/output request indices and IOSurfaces with
these logical buffers; retain IOSurface sizes and row/plane properties if traced.

For captured addresses, preserve the original value **and** its owning buffer
plus relative offset. Linux must allocate new buffers and relocate selectors or
offsets; macOS IOVAs/physical addresses cannot be reused. Validate every DMA
extent and each enabled coefficient slot against its payload. The Linux ABI
needs `tsk_size`, `td_count`, first `td_size`, `handles[32]` and `btsp_handle`.
Its driver synthesizes the coefficient BAR from the command-buffer base and
aligned command length; confirm this fits the captured container before replay.

### Live kernel/driver registers — conditional tracing task

If the executable/bindings are missing, replay fails, or correct replay is
still much slower, trace **one warmed head4096 and one warmed fused-FFN call**
on macOS first, paired with the equivalent Linux submissions. Record state
immediately before submission and at completion, correlated with case/input/
program hashes and timestamps. Extend to attention after these work.

Needed state:

- Task Manager (M1 base `0x20000`): submitted address/info/push values, queue
  enable, committed/status/error fields, completion event info/timestamp and
  interrupt enable/ack state. Capture completion-event reads through the driver
  because reads can consume events.
- Task Queue (M1 base `0x21000`, queue stride `0x148` in the current Linux
  driver): actual queue/priority, both bank sets of BAR entries, descriptor
  address/size, NID, status/vacancy/info and bootstrap descriptor. Also record
  task counts, cache maintenance and synchronization around submission.
- DART/IOMMU mappings: relevant buffer IOVA ranges, mapped lengths, page size,
  fault/status and mapping identity. Retain buffer-relative offsets so mapping
  differences can be separated from program differences.
- ANE power/clock context: actual performance state/frequency when exposed,
  power-domain and clock-gating state, active-engine mask, warmup and idle
  behavior. Pair device timing, enqueue-to-completion and host wall time. Treat
  a zero `powermetrics` ANE-power reading as uninformative by itself.

The current macOS script does **not** collect these live registers. This needs
an available driver trace/debug interface or additional instrumentation; record
the interface and unavailable fields in the report. There is no supplied
generic macOS MMIO-dump command. Prefer recording existing driver accesses;
do not probe arbitrary addresses or change clock/power/register programming
as part of this capture task. The offset names above are grounded in the
surveyed base-M1 Linux driver and must be checked against the traced runtime.

## 5. Package and return the evidence

```bash
.asahi/venv-metal/bin/python .asahi/m1-sd-macos/tools/check_manifest.py \
    .asahi/m1-sd-macos --record-capture
tar -czf .asahi/m1-sd-macos-results.tar.gz -C .asahi m1-sd-macos
shasum -a 256 .asahi/m1-sd-macos-results.tar.gz \
    > .asahi/m1-sd-macos-results.tar.gz.sha256
```

Return the archive and digest to this repo's `.asahi/` on Linux via a shared
disk or file transfer. Git pull carries source, fixtures and reports; generated
weights/programs/captures are ignored and need this separate transfer. Include
macOS build, chip, compiler/framework versions, Python packages, Git revision,
all failure logs and `capture-manifest.json`. Compare only within the recorded
OS/compiler version; recompiling elsewhere can produce different valid bytes.

Write `notes/m1_macos_capture_report.md` with mean/max Metal rates, per-width
verify latency/identity, per-graph ANE latency and compile/evaluate/export
status, numerical differences and exactly which program/bindings/register
fields were obtained or remain missing. Keep small result JSONs under
`notes/asahi/`; reference large artifacts by archive SHA256 and relative path.

## 6. Implement and verify the Asahi path after capture

1. Verify bundle hashes and H13G architecture. Decode task chains and compare
   evaluated/offline programs, coefficient layouts and bindings. Resolve any
   missing loaded executable before treating offline HWX as the same program.
2. Implement bounded allocation, BAR relocation, coefficient/constant loading,
   scratch reuse and bootstrap submission **inside this repo**. Start with
   head4096, then head8192 and fused FFN, followed by RMSNorm and attention.
   Preserve dependencies and task order; do not assume the linear-only template
   covers fused programs.
3. Replay identical inputs on Linux against macOS `output-0.bin`, checking all
   physical rows, finite/unwritten output, max error and relative RMSE. Repeat
   across all four prompts. Explain precision differences rather than choosing
   tolerances solely to pass a failing case.
4. Measure warmed device/submit/host times, transfer bytes and submissions.
   Compare like-for-like graphs and inputs with macOS. Replace native per-op
   execution only after numerical replay passes. Keep coefficients and scratch
   resident; avoid CPU round trips between tasks within a fused graph.
5. Integrate into the draft and benchmark fresh-cache SD: four prompts, two
   passes, 100 new tokens, bf16 target, greedy baseline, mean/max paired speedup,
   acceptance/cycle, phase times and token identity. Fix small-block Vulkan
   verification with the matched Metal evidence; it remains a separate GPU
   bottleneck even if the draft becomes faster.

Then request additional **M1-generated** dumps for whole-draft graphs, RoPE/QK
norm in graph, accumulating/sliding KV state (including reset, commit and
rollback behavior), and complete vocabulary heads. Capture LUT6 coefficient
payloads/palettes and decoder register fields only if M1 compilation and
placement succeed; verify quality using real hidden states. Target-layer chunks
require their own M1 shapes/state captures and an 8 GB memory budget. These
later captures are not included in the first 20-case kit.

## Completion criteria

- [ ] Matched Metal target benchmark and all width probes saved, including
  mismatches, model/package hashes and same-M1 hardware identity.
- [ ] All 20 ANE cases attempted; successful evaluated outputs and artifacts,
  failure logs and independent export status retained.
- [ ] H13G task/register streams, coefficients and bindings decoded for the
  graphs selected for replay; evaluated/offline provenance resolved.
- [ ] Live tracing collected where needed, with unavailable fields stated.
- [ ] Bundle returned to Linux and report written with archive SHA256.
- [ ] Linux replay passes all four real inputs before SD integration.
- [ ] End-to-end results report correctness, mean/max speedup and remaining
  bottlenecks. Reproducing the original M4 Pro 2.21× is not a completion claim
  for this smaller M1 setup without its own measured result.
