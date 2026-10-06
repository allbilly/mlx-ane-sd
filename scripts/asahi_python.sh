#!/usr/bin/env bash
# Use the isolated Vulkan MLX environment and its optional local Fedora libs.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$repo_dir/.venv-asahi/native/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$repo_dir/.venv-asahi/bin/python" "$@"
