# M1 macOS DFlash 3× reproduction attempt

Machine: base Apple M1 MacBook Air, 8 GB. Target remains the pinned `mlx-community/Qwen3-0.6B-bf16`; public trained draft: [`orestis-z/dflash-qwen3-0.6b-microcycle-dflash`](https://huggingface.co/orestis-z/dflash-qwen3-0.6b-microcycle-dflash).

Four unchanged repo prompts × 2 alternating-order passes × up to 100 tokens. Each generation starts with fresh caches. Draft features remain resident in MLX. Decode time includes drafting, context updates, verification and cache rollback. Stock mlx-lm uses greedy sampling and the same 32-token prefill chunks. Stock generation_tps excludes the first token; the custom loop includes its forward. The separate scalar-loop baseline and both token sequences are retained in every receipt.

A configuration qualifies only when all SD tokens equal scalar greedy decoding and that baseline also equals stock mlx-lm. Reported speedups below use stock mlx-lm.

| Configuration | Stock baseline tok/s | SD tok/s | Mean paired speedup | Max trial | SD identity | Stock identity | Qualified |
|---|---:|---:|---:|---:|---:|---:|---|
| p7_bf16_stock | 45.66 | 26.61 | 0.58× | 0.76× | 2/8 | 8/8 | no |
| p3_bf16_stock | 45.46 | 26.19 | 0.58× | 0.71× | 2/8 | 8/8 | no |
| p1_bf16_stock | 46.86 | 24.84 | 0.53× | 0.58× | 2/8 | 8/8 | no |
| p7_bf16_small_metal | 46.07 | 33.31 | 0.72× | 0.90× | 4/8 | 8/8 | no |
| p7_fp16_draft_small_metal | 46.25 | 33.88 | 0.73× | 0.91× | 4/8 | 8/8 | no |
| p7_bf16_serial_reference | 45.98 | 26.20 | 0.57× | 0.61× | 8/8 | 8/8 | yes |

**3× mean reproduced:** no.

Best configuration passing all token checks: **p7_bf16_serial_reference**, **0.57× mean**, **0.61× best trial** versus stock mlx-lm. All 800 generated token IDs match both greedy references.

| Prompt | Stock tok/s | SD tok/s | Paired mean | Max trial | Tokens/cycle |
|---|---:|---:|---:|---:|---:|
| capital | 45.49 | 25.46 | 0.56× | 0.57× | 1.80 |
| fibonacci | 45.41 | 27.30 | 0.60× | 0.61× | 2.48 |
| math | 46.74 | 26.33 | 0.56× | 0.57× | 2.20 |
| story | 46.28 | 25.71 | 0.56× | 0.56× | 1.94 |

## Measured bottleneck

| Configuration | Tokens/cycle | Verify ms/cycle | Context ms/cycle | Draft body ms/cycle | Draft head ms/cycle | Zero-draft-cost bound vs stock |
|---|---:|---:|---:|---:|---:|---:|
| p7_bf16_stock | 2.02 | 55.16 | 1.84 | 5.38 | 13.24 | 0.80× |
| p3_bf16_stock | 1.95 | 54.01 | 1.82 | 5.20 | 13.33 | 0.79× |
| p1_bf16_stock | 1.61 | 52.66 | 1.79 | 5.02 | 5.44 | 0.65× |
| p7_bf16_small_metal | 2.02 | 46.54 | 1.64 | 4.75 | 7.65 | 0.94× |
| p7_fp16_draft_small_metal | 2.02 | 45.22 | 2.78 | 4.60 | 6.80 | 0.97× |
| p7_bf16_serial_reference | 2.10 | 60.26 | 1.56 | 5.11 | 13.24 | 0.76× |

The draft commits 1.61–2.10 tokens per cycle across configurations. Batched verification alone takes 45–55 ms per cycle, versus about 22 ms per token for stock greedy MLX. Drafting and context updates add 12–20 ms per cycle. The diagnostic bound holds the observed acceptance and verification timings fixed and removes all other cycle work; it is not a measured speedup or a universal hardware ceiling. Its highest mean value is 0.97×.

The sweep runs all seven cache/kernel checks before timing. It retained 48 paired trials. The identity columns distinguish SD-versus-scalar checks from scalar-versus-stock checks.

Serial verification matched both greedy references on every trial. The failures with batched verification isolate the token drift to batched target numerics for this checkpoint, rather than rejected-cache rollback.

The optional small-block Metal kernel keeps bf16 target weights and activations and accumulates dot products in FP32. Its reduction order differs from stock MLX; passing the token checks is required. fp16 variants change only the drafter. They still verify with the bf16 target. Failed numerical variants are retained and excluded from the qualified best result.

This smaller-model M1 experiment is separate from the README's M4 Pro/64 GB Qwen3-4B full-ANE stack. The 4B reproduction recorded about 10 GB peak memory. These runs measure MLX GPU drafting and verification, including an experimental Metal kernel; they do not establish a full-ANE result on M1.

Reproduction (reuse the downloaded, pinned model files):

```bash
.asahi/venv-metal/bin/python -u scripts/run_macos_dflash_sweep.py
```

Model downloads, when needed, use the Hugging Face CLI:

```bash
hf download mlx-community/Qwen3-0.6B-bf16 \
  --revision 42096995f6402fde107068cf530136fe64b604f8 \
  --local-dir .asahi/models/qwen3-0.6b-bf16-metal
hf download orestis-z/dflash-qwen3-0.6b-microcycle-dflash \
  --revision 4dd1e04078f993593338ef1f9403179e41e4580e \
  --local-dir .asahi/models/microcycle-dflash --include '*.json' '*.safetensors'
python3 - <<'PY'
import json
from pathlib import Path
Path('.asahi/models/microcycle-dflash/SOURCE.json').write_text(json.dumps({
    'model': 'orestis-z/dflash-qwen3-0.6b-microcycle-dflash',
    'revision': '4dd1e04078f993593338ef1f9403179e41e4580e'}, indent=2) + '\n')
PY
.asahi/venv-metal/bin/python scripts/run_macos_dflash_sweep.py \
  --target .asahi/models/qwen3-0.6b-bf16-metal
```

Receipts:

- [p7_bf16_stock](m1_macos_dflash_sweep_p7_bf16_stock.json)
- [p3_bf16_stock](m1_macos_dflash_sweep_p3_bf16_stock.json)
- [p1_bf16_stock](m1_macos_dflash_sweep_p1_bf16_stock.json)
- [p7_bf16_small_metal](m1_macos_dflash_sweep_p7_bf16_small_metal.json)
- [p7_fp16_draft_small_metal](m1_macos_dflash_sweep_p7_fp16_draft_small_metal.json)
- [p7_bf16_serial_reference](m1_macos_dflash_sweep_p7_bf16_serial_reference.json)

All existing Asahi source files and the archived capture kit remain unchanged.
