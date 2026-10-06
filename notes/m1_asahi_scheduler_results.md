# M1 Asahi: CPU scheduling and useful-row readback

The confirmation benchmark over four prompts and four passes measures
**61.39 tok/s SD versus 36.22 tok/s stock MLX BF16: 1.695×**. It uses
100-token warmups and 100-token
measured generations. Every SD trial matches the macOS compressed-target
generation and acceptance/cache decisions. This exceeds the matched M1 macOS
ratio of **1.593×** (72.00 / 45.19 tok/s).

The **0.776×** result was the initial Linux replay, before transport tuning.
macOS did not record that slowdown. The M1 kit uses Qwen3-0.6B and LUT6; the
original M4 Pro / Qwen3-4B result is a separate experiment.

## What changed

1. Target and draft K/V remain resident. Commits upload only accepted rows;
   output readback copies only those rows. Reset and compiler padding remain
   deterministic, and stale output handles are rejected.
2. A compiled NEON helper reduces FP16 vocabulary outputs directly from ANE
   mappings, preserving finite-value checks, signed-zero ties and the lowest
   token ID. Target verification stops reading at the first rejected candidate
   or the bonus token. Unused predictions never affect emitted tokens or cache
   state. The trained seven-token proposal and all five ANE programs remain
   unchanged.
3. The process and its workers use P-cores 4–7. `--cpu-util-min 1024` requests
   the Linux scheduler's maximum utilization minimum for the benchmark thread
   and subsequently created workers. It retains the scheduling policy, nice
   value and other parameters. The same hint and affinity apply to MLX, ANE AR
   and ANE SD. It changes no system governor, kernel module or other process.
4. All configurations warm every prompt for 100 tokens. MLX is synchronized
   between trials outside its generation timer. This avoids carrying queued
   work into ANE trials. Warmups, loading and prefill are excluded from rates.

## Measurements and controls

| Configuration / stage | SD tok/s | MLX tok/s | Ratio of means |
|---|---:|---:|---:|
| Initial replay | 26.96 | 34.72 | 0.776× |
| Resident cache + mapped reduction | 39.50 | 35.19 | 1.123× |
| Native reduction alone | 42.91 | 35.20 | 1.219× |
| Useful-row reads, P-cores, full warmups | 54.52 | 35.09 | 1.554× |
| Four readback workers, full warmups | 44.53 | 35.24 | 1.263× |
| **One worker + utilization hint, full warmups** | **61.06** | **35.23** | **1.733×** |
| **One-worker confirmation, four passes** | **61.39** | **36.22** | **1.695×** |

The earlier P-core trial with ten-token warmups gave 1.627×, but stronger
warmups reduced it to 1.554×. Shortening lookahead to four gave 1.500×.
Neither result is used to claim the macOS ratio was reproduced.

In an isolated probe on real mapped vocabulary outputs, four workers reduced
seven rows in 2.85 ms versus 8.66 ms with one worker. Wider reads alone did not
improve performance. In a complete benchmark without the utilization hint,
parallel readback accompanied slower target and draft call spans, outweighing
the isolated savings. With one worker and the hint, target call spans decreased
from about 7.6–7.8 to 7.0 / 6.9 ms, and padded ANE AR increased to 51.67 tok/s.
These are host-call measurements; they do not isolate ANE clock frequency or
pure driver overhead.

| Prompt | MLX BF16 tok/s | Padded ANE AR tok/s | ANE SD tok/s | SD / MLX |
|---|---:|---:|---:|---:|
| capital | 36.54 | 51.90 | 64.70 | 1.770× |
| fibonacci | 35.59 | 52.00 | 68.98 | 1.938× |
| math | 36.27 | 51.93 | 61.39 | 1.693× |
| story | 36.48 | 51.91 | 50.50 | 1.384× |
| **Mean** | **36.22** | **51.94** | **61.39** | **1.695×** |

Mean paired speedup is **1.696×**, maximum paired speedup **1.954×**, and
maximum SD trial **69.12 tok/s**. SD / the same padded ANE AR control is
**1.182×**. Linux's relative SD/MLX result is higher than macOS's; its absolute
SD rate remains lower than macOS's 72.00 tok/s. The Vulkan MLX baseline also
remains lower than macOS's 45.19 tok/s Metal baseline.

[Four-pass confirmation](m1_linux_full_ane_bench_uclamp_confirm_warm100_20261006.json),
[initial one-worker benchmark](m1_linux_full_ane_bench_uclamp_pcores_warm100_20261006.json),
[isolated mapped-read probe](asahi/m1_native_mapped_read_probe_20261006.json),
[full-warmup P-core control](m1_linux_full_ane_bench_grouped_pcores_warm100_20261006.json),
[four-worker control without hint](m1_linux_full_ane_bench_burst_omp4_warm100_20261006.json).

[Derived validation and summary](asahi/m1_full_ane_scheduler_summary_20261006.json).

The [four-worker extended run with the hint](m1_linux_full_ane_bench_uclamp_omp4_warm100_20261006.json)
also passed all correctness gates. It averaged 55.41 / 27.59 tok/s (2.008×),
but both stacks slowed during parts of the run: MLX fell to about 17 tok/s and
SD from roughly 67–75 to 29–39 tok/s. Its aggregate ratio is preserved rather
than used as the confirmed headline. CPU quota counters showed no throttling;
the cause of that system-wide slowdown remains unresolved. The subsequent
one-worker confirmation has MLX rates of 35.38–36.98 tok/s across all sixteen
trials, with every prompt faster under SD. It records frequency, system power,
temperature and pressure observations outside the generation timers.

The [instrumented one-worker profile](asahi/m1_full_ane_profile_uclamp_prefix_20261006.json)
also reproduces all four SD traces. Target chunks average 7.00 / 6.92 ms per
call, with synchronous submit spans of 6.60 ms. Submission consumes 1.34 / 1.29
ms of the calling thread's CPU time within those wall spans; this does not
isolate device execution from kernel waiting. Head reduction averages 5.43 ms
across target and draft calls. Target K/V prefix reads average 0.33 / 0.32 ms
per output, and accepted-cache updates 0.38 / 0.36 ms per chunk. These profile
statistics include prefill and are separate from throughput measurements.

## Correctness and scope

Before timing, the benchmark regenerates all four complete macOS SD traces,
checks **72 selected calls / 588 output hashes**, hashes actual resident input
surfaces including padding, compares **386 head decisions** with full readback,
and checks **216 K/V prefixes** against full outputs. All sixteen measured SD
trials / **1,600 tokens** in the confirmation match the same LUT6 ANE AR target
and macOS generations, including the acceptance and cache decisions. The
initial two-pass result independently checks another 800 generated tokens.

All **19 host tests** pass. The native reducer tests cover every finite FP16
value, negative and positive zero, ties across vocabulary chunks, odd widths,
strides, nonfinite outputs and the wide-row fallback. Cache tests cover partial
commits, reset, invalid updates and stale mappings.

The target remains LUT6 compressed, so BF16 token identity is not claimed.
The ANE AR control computes padded B=8 blocks. A true B=1 ANE control still
needs new compilation. Offline HWX outputs match the macOS CoreML goldens;
the exact executable loaded internally by CoreML remains unavailable. New
macOS dumps were not needed for these host transport and scheduling changes.

## Reproduce the one-worker result

The driver must be loaded and `/dev/accel/accel0` accessible. A C compiler with
OpenMP is required; the native helper builds into ignored `.asahi/build/` and
records compiler flags, source and library hashes in the receipt.

```bash
replay_target=/home/asahi/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/42096995f6402fde107068cf530136fe64b604f8

scripts/asahi_python.sh scripts/run_m1_full_ane_replay.py \
  artifacts/m1-full-ane --mode bench --target "$replay_target" \
  --draft .asahi/models/microcycle-dflash \
  --transport resident --head-readback native \
  --verify-readback accepted-prefix --kv-readback prefix \
  --native-threads 1 --cpu-affinity 4,5,6,7 --cpu-util-min 1024 \
  --warmup-new 100 --max-new 100 --repeats 4 \
  --out notes/m1_linux_full_ane_bench_uclamp_new.json
```

Use a fresh receipt path. Default warmups now run up to 100 tokens. The
historical `resident` / `mapped` / full-output transport remains available;
the optimized recipe explicitly selects native and prefix readback. Both
`~/ane.lock` and `~/gpu.lock` are held across verification and all trials.
