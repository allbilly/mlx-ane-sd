#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -s)" != Darwin || "$(sysctl -n machdep.cpu.brand_string)" != 'Apple M1' ]]; then
  echo 'This capture kit requires macOS on the base Apple M1.' >&2
  exit 2
fi
kit_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$kit_dir/manifest.json" ]]; then
  echo 'Run scripts/prepare_m1_sd_capture.py first; see task.md in the repository.' >&2
  exit 2
fi
mkdir -p "$kit_dir/build"
xcrun clang -O2 -fobjc-arc -fblocks -I"$kit_dir/Orion/core" \
  -framework Foundation -framework IOSurface \
  "$kit_dir/tools/capture.m" "$kit_dir/Orion/core/ane_runtime.m" \
  "$kit_dir/Orion/core/iosurface_tensor.m" -o "$kit_dir/build/capture"
sw_vers > "$kit_dir/build/macos.txt"
sysctl -n machdep.cpu.brand_string > "$kit_dir/build/chip.txt"
xcrun clang --version > "$kit_dir/build/compiler.txt"
python3 "$kit_dir/tools/check_manifest.py" "$kit_dir"
status=0
for case_dir in "$kit_dir"/cases/*; do
  if ! "$kit_dir/build/capture" "$case_dir" > "$case_dir/macos.log" 2>&1; then
    echo "Capture failed or HWX export incomplete: $case_dir" >&2
    status=1
  fi
done
python3 "$kit_dir/tools/check_manifest.py" "$kit_dir" --record-capture
exit "$status"
