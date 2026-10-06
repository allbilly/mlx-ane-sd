# Matched M1 macOS capture report

The macOS ANE capture has finite evaluated outputs for all **20 graph/prompt
pairs** after two changes confined to generated copies of the runner and
fixtures under ignored `.asahi/`. All tracked Asahi and macOS source files are
unchanged. The original run's eight failures and the unsuccessful packing
trial are retained alongside successful follow-ups.

The pinned target was downloaded through the installed `hf download` CLI and
all target hashes were verified. The matched Metal benchmark completed.

## Machine and inputs

- Base Apple M1, 8 GB, arm64, on AC power.
- macOS 27.0.1, build 26A434; Apple clang 21.0.0.
- Repository revision: `bb4617f76df00fc66d1b4faef3ec638e7912559b`.
- Draft: `orestis-z/dflash-qwen3-0.6b-microcycle-dflash`, revision
  `4dd1e04078f993593338ef1f9403179e41e4580e`; downloader verified its four hashes.
- Target: `mlx-community/Qwen3-0.6B-bf16`, revision
  `42096995f6402fde107068cf530136fe64b604f8`.
- Four real input fixtures from `asahi/fixtures/m1-sd/`; preparation verified
  their pinned hashes. No Linux ANE backend was built or executed on macOS.
- Python 3.11; `mlx==0.32.2`, `mlx-lm==0.31.3`; full package list in the bundle.

## Matched Metal result

Four prompts × two passes, up to 100 new tokens each: **38.95 tok/s mean**, **40.35 tok/s max**. Greedy token identity to the recorded Linux traces: **2/8 trials**. Token IDs, timings and all mismatches are retained in [m1_macos_metal_reference.json](asahi/m1_macos_metal_reference.json).

| Prompt | Mean decode tok/s | Linux token identity |
|---|---:|---:|
| capital | 37.76 | 0/2 |
| fibonacci | 39.43 | 2/2 |
| math | 39.68 | 0/2 |
| story | 38.93 | 0/2 |

Fixed teacher-history probes include target hidden-feature transfer. Width-one and block predictions are compared within macOS. Linux columns use the stock route from `notes/asahi/sd_routes.json`.

| Width | Metal mean median ms | Metal max median ms | Metal block/scalar identity | Linux stock mean median ms |
|---:|---:|---:|---:|---:|
| 1 | 25.93 | 28.10 | 4/4 | 48.00 |
| 2 | 52.72 | 55.42 | 4/4 | 166.08 |
| 4 | 53.50 | 53.77 | 4/4 | 162.62 |
| 8 | 54.45 | 54.81 | 4/4 | 163.70 |
| 16 | 53.68 | 54.39 | 4/4 | 159.53 |
| 32 | 57.76 | 59.16 | 3/4 | 164.70 |

This is a matched target/verification measurement, not a macOS DFlash SD loop.

## ANE results

Each successful capture used two warmup calls and 20 timed evaluations.
Numbers below are the mean and maximum of the four per-prompt median **host
call latencies**, not hardware-only timing. Physical padded rows are included
in output checks. Host diagnostics use FP32 arithmetic with FP16 projection
boundaries; they are not full-model token-identity tests.

| Graph | Outputs | Mean median ms | Max median ms | Max relative RMSE vs host |
|---|---:|---:|---:|---:|
| RMSNorm | 4/4 | 0.115 | 0.178 | 0.1223% |
| Fused FFN | 4/4 | 0.456 | 0.471 | 0.1574% |
| Cached attention | 4/4 | 0.166 | 0.204 | 0.0592% |
| Head 4096 | 4/4 | 0.260 | 0.316 | 0.0195% |
| Head 8192 | 4/4 | 0.404 | 0.454 | 0.0192% |

Detailed per-case receipts, original errors, selected golden paths and output
hashes are in [m1_macos_ane_capture.json](asahi/m1_macos_ane_capture.json).

The unmodified capture runner succeeded for 12/20 cases: all RMSNorm and both
head sizes. All four attention requests failed with an IOSurface size error.
All four fused FFN requests failed with `verifyBundleAtPath: invalid model`.
Independent offline export succeeded for all 20 original graphs.

Successful generated follow-ups:

1. Attention input request indices follow the compiler's **k, q, v** order.
   The original descriptor used **q, k, v**. Reordering the request surfaces
   fixed all four attention evaluations; tensors and graph coefficients stayed
   identical.
2. The FFN's four coefficient files were packed into a single BLOB. Both the
   MIL BLOB record offsets and each record's **absolute payload offset** were
   relocated. Corrected packing fixed all four evaluations. Every offline
   coefficient segment SHA256 matches the original export. The first packing
   trial omitted the payload relocation and produced wrong/nonfinite outputs;
   that trial is retained and excluded from selected goldens.

The generated source is `build/capture-sorted-inputs.m`. Trial descriptors
under `supplemental/` record the input/weight changes. Additional request-surface
captures for the original 12 successes produced byte-identical outputs.
`ane-summary.json` selects exactly 20 goldens; unsuccessful trials are not
selected.

## Dumps obtained and remaining gaps

The bundle preserves all original cases, source MIL, real inputs, FP16 BLOBs,
host diagnostics, logs, evaluated output buffers, runtime temporary directories,
offline HWX exports, compiler status dictionaries and successful follow-ups.

For all 20 original offline exports, decoding validated **H13G, CPU subtype 4,
ISA v7** and followed variable-sized descriptor chains. The originals contain
332 tasks and 41,232 ordered register writes. Decoded directories contain:

- `container.json`: complete raw load commands, segments, sections, thread/BAR
  data, symbols and relocations.
- `task-descriptors.bin`, `tasks.json`: original descriptor bytes, full headers,
  packet order, next pointers/sizes, flags and base selectors.
- `registers.json`: task ID, packet/word order, raw register value and engine
  byte offset; unknown words and padding are preserved.
- `bindings.json`, `payloads/`: compiler BAR values and original segment data,
  including complete packed coefficient segments. New follow-up decodes also
  expose tensor shapes/strides and compiler BAR segment ownership.

The local `~/ane/gpt2/hwx.py` reader independently checked **51 original and
follow-up exports**. All task chains, ten-word headers, final register states
and compiler BAR tables match the capture decoder. The comparison retains our
ordered stream as the authority for repeated writes. `ane-parser-crosscheck.json`
records every program and hash. A licensed copy of that reader and its pinned
provenance are included under `build/ane-reference/`; the source repository was
only read.

Additional `register-fields.json` reports annotate the seven H13 command blocks.
Raw values and unknown writes remain in `registers.json`. Two reference-map
disagreements are recorded explicitly: source DMA format is at `0x13838` in
the Python reader and C struct, while the Markdown table says `0x138A4`;
coefficient DMA cache-hint bit ranges differ, so those bits remain undecoded.
The local macOS exporter also performs separate offline compilation; its trace
script uses Linux DRM ioctls and does not provide macOS runtime readback.

All selected cases have a request-surface capture with IOSurface IDs, allocation
sizes, row/plane properties, request indices, initial input/output buffers and
logical shapes/strides. These are userspace request observations, not device
IOVAs or live Task Queue BAR readings.

**The executable used by the evaluated runtime remains missing.** Its copied
`net.plist` is MIL source, and `data` is an input coefficient BLOB. No loaded HWX
was found in those temporary directories. Consequently, the offline HWX must
not be claimed as the executable that produced the golden, even where separate
original/follow-up exports have identical coefficient bytes.

Read-only runtime metadata probes for a warmed capital head4096 and fused FFN
call retained input/output symbol indices, live tensor shapes/strides, model
state, queue depth, program and intermediate-buffer handles.
`supplemental/userspace-observation/` includes before-submit and after-completion
object snapshots, host monotonic timestamps, and MIL/input/blob/output/offline
program hashes. The interface is owned Objective-C object metadata; these are
not kernel register traces. No exposed NSData contained a loaded HWX.

Xcode Instruments 27.0 also recorded **head4096 and fused FFN** in separate
attached-process runs. Each recording contains compile/load events, all 22
prediction intervals (two warmups plus 20 timed calls), and raw kdebug tables.
Both evaluated outputs are byte-identical to the selected goldens. The trace
bundles and XML exports are under `supplemental/instruments-attached/`;
[m1_macos_instruments.json](asahi/m1_macos_instruments.json) records event
timestamps, input/program/output hashes and observed interval durations.
These are diagnostic traces, excluded from the four-prompt performance table.
They expose execution intervals, but not the requested MMIO, device IOVAs,
runtime BAR relocation or loaded binary. The initial Instruments launch attempt
exited before fixture loading; its trace and error logs are preserved. Attaching
to a process launched from the working environment succeeded.

DTrace probe enumeration failed because additional privileges are required;
noninteractive `sudo` required a password. SIP is enabled. Those failures and
the installed Instruments/template inventories are retained under `build/`.

IORegistry identifies `H11ANEIn` / `com.apple.driver.AppleH11ANEInterface` with
architecture `h13g`, subtype 4 and 16 cores. The inventory is saved under
`build/`. No supplied driver interface exposes the requested submission and
completion MMIO trace. Live Task Manager/Queue registers, DART IOVAs, runtime BAR
relocations, initial internal scratch contents/aliases and clock/power-state
registers were **not obtained**. No arbitrary register addresses were probed.
Linux replay, relocation validation and end-to-end SD integration remain
separate follow-up work. These microbenchmarks do not establish SD speedup.

## Historical bundle (deleted)

The capture used working directory `.asahi/m1-sd-macos/` and archive
`.asahi/m1-sd-macos-results.tar.gz`, with a digest file beside it. Both were
deleted during cache cleanup after full-stack Linux validation. This report
and small receipts remain under `notes/`. The archive included original failures and successful follow-ups,
selected goldens, the pinned target, full decoded programs, coefficient data,
request surfaces, package/build metadata and hash manifests.

Archive SHA256: `7862a6b32164023818dc47fea18aeba8a633f2a74d7da651af33b1894bed42d2`.

Archive integrity verification passed before deletion: all **2,430 files**
matched their recorded hashes, and the archive SHA256 matched its sidecar. See
[m1_macos_archive_verification.json](asahi/m1_macos_archive_verification.json).

## Historical microkernel reference (deleted)

The selected 20 successful cases were packaged at
`.asahi/compact-m1/archived-microkernel-reference/`, outside Git. Its three
compressed, deduplicated pack files totaled **44.9 MB**, reconstructing 433 captured files:
20 complete offline HWX programs with embedded coefficients, descriptor and
register streams, compiler bindings, real inputs, surface observations, and
20 macOS golden outputs. Segment payload copies are regenerated from the HWX
and verified against their recorded hashes.

All reconstructed files and regenerated segments were checked against the
original capture. The optional bundle and full archive have now been deleted;
neither is required by the full-stack runner. After Git pull on Asahi, use the
[compact full-stack kit](../artifacts/m1-full-ane/README.md) and its pinned public
checkpoints for reconstruction and replay.
This focused reference did not include the BF16 target, original source-weight
BLOBs, failed trials, Instruments recordings or duplicate runtime temporary
directories. The exact runtime-loaded HWX and live device data remain missing,
and these 20 historical microkernels were not verified on Linux. The separate
full-stack kit has since passed Linux replay and measured a **1.695×** mean
speedup; see [the confirmed Linux recipe](m1_asahi_scheduler_results.md).
The useful request-surface diagnostic source survives as a small
[patch](m1_capture_request_surfaces.patch). See
[cache recovery instructions](m1_full_ane_asahi_handoff.md#recreate-local-assets-only-when-needed)
and [the cleanup receipt](m1_local_cache_cleanup.json).
