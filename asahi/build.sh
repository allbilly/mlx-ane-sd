#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/.asahi/build"
"${CC:-gcc}" -O3 -fopenmp -march=native -fPIC -shared -DQWEN3_USE_ANE \
  -I"$repo_dir/asahi/vendor/qwen3.c" \
  "$repo_dir/asahi/vendor/qwen3.c/ane/ane_matmul.c" -lm \
  -o "$repo_dir/.asahi/build/libane_sd.so"
