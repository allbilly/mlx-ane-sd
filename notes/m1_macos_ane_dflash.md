# M1 macOS: native ANE DFlash + MLX target

The separate ANE runtime works, but this experiment did not reproduce a 3× speedup. MLX itself runs on CPU/GPU; the draft calls ANEForge's native e5rt runtime with ANE-only device mask `0x4`. The bf16 target remains on MLX/Metal.

Hardware: base M1 MacBook Air, 8 GB, macOS 27.0.1. Target: pinned `mlx-community/Qwen3-0.6B-bf16`. Draft: public `orestis-z/dflash-qwen3-0.6b-microcycle-dflash`, converted from bf16 to fp16 for ANE. Three draft layers, 8-token trained block, seven proposed tokens, fixed context capacity 256. The full vocabulary ANE head is tiled at 8,192 columns; no reduced-vocabulary approximation.

Four repository prompts × two passes × 100 tokens per configuration. Baseline/SD order alternates between passes. Both device locks are held for the complete sweep. All paths use stock mlx-lm's recommended wired-memory limit. Loading, compilation and prefill are excluded; drafting, transfers, verification, cache commits and rollback are included in decode timing.

| Configuration | Stock MLX mean tok/s | SD mean tok/s | Mean paired speedup | Best trial | Matching greedy trials |
| --- | ---: | ---: | ---: | ---: | ---: |
| body_ane_head_gpu_batch | 44.02 | 12.95 | 0.29× | 0.39× | 2/8 |
| body_ane_head_ane_batch | 42.68 | 14.03 | 0.33× | 0.46× | 2/8 |
| body_ane_head_ane_serial | 26.09 | 16.47 | 0.66× | 1.02× | 8/8 |

The serial-verification configuration matches all **800/800 generated token IDs** against both scalar greedy decoding and stock mlx-lm. Batched GPU verification changes near-tie argmax results on three prompts and is retained as a numerical variant, not a byte-identical reproduction.

| Configuration | ANE body ms/cycle | Head ms/cycle | Context/transfer ms/cycle | GPU verify ms/cycle | Native ANE calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| body_ane_head_gpu_batch | 9.90 | 27.20 | 0.68 | 119.35 | 398 |
| body_ane_head_ane_batch | 10.14 | 19.57 | 0.74 | 115.80 | 796 |
| body_ane_head_ane_serial | 12.84 | 28.57 | 1.59 | 89.20 | 752 |

## Correctness and placement

Each compiled draft passed real-input comparison with the MLX fp16 reference on all four prompts (minimum hidden-state cosine similarity 0.99999148). The first proposed token matched the reference on every prompt. Both native program constructors reject a device mask other than `0x4`; successful native evaluation counts are retained for every trial. The body fuses 263 graph operations. ANE-head variants execute the full vocabulary through a second native program.

The initial graph returned zero hidden states: ANEForge's FP16 `reduce_l2_norm` overflowed on real projector/residual values reaching about 8,000. Power-of-two rescaling before normalization fixed the issue. Zero-output runs are marked invalid in `m1_macos_ane_dflash_zero_output.*`; the superseded unwired control is retained in `m1_macos_ane_dflash_unwired.*`. These are not speedup evidence.

## Limits

Stock MLX rates ranged from 19.63 to 46.50 tok/s during the desktop session. Exclusive GPU/ANE reservations do not eliminate background CPU activity, paging or clock variation. These are paired measurements of this implementation, not an isolated hardware ceiling or a claim that M1 ANE strength alone explains the result. Matching the wired-memory setting did not recover batched verification performance.

This tests an ANE draft with a GPU target. The README's best result also offloads the much larger Qwen3-4B target to ANE on an M4 Pro/64 GB machine. That full-ANE stack has not been reproduced here. Decode timing includes the first target forward after prompt prefill; stock mlx-lm's own timing convention differs by roughly one token at this generation length.

Existing Asahi source files and reference repositories were not modified.

## Reproduce

```bash
cd /Users/yeren/Desktop/mlx-ane-sd
.asahi/venv-metal/bin/python -u scripts/bench_macos_ane_dflash.py
.asahi/venv-metal/bin/python scripts/report_macos_ane_dflash.py
```

Uses the already downloaded, pinned HF target and public draft plus the existing built ANEForge runtime in `~/Desktop/ANEForge`. Source hashes, model revisions/weight hashes, runtime library hash, host states, token IDs and per-cycle traces are in [the raw receipt](m1_macos_ane_dflash.json). See [the reference survey](m1_macos_ane_reference_survey.md) for MLX/Core ML/direct-ANE distinctions.
