# Vendored native M1 ANE backend

Copied from `https://github.com/allbilly/qwen3.c` at the revision recorded in
[UPSTREAM.json](UPSTREAM.json). The manifest records the **original** SHA256
of every copied file. The MIT license is retained in [LICENSE](LICENSE).
The copy is independent of `~/qwen3.c`; editing or building here changes no
other repository.

`ane/ane_matmul.c` and `.h` have a local extension:
`ane_plan_create_fp16` accepts a row-major `[outputs, inputs]` binary16 matrix
without first requantizing it to Q8. Allocation and register patching are
shared with the existing Q8 entry point. The resident weight packing,
synchronous submission, and finite-output checks are retained.

`ANE_PROFILE=1` enables diagnostic clock counters for input packing, upload,
synchronous submission, download and output unpacking. `ane_device_profile`
reads these counters; `ane_device_profile_reset` clears them. Timing is disabled
by default and does not change matrix arithmetic. The submission counter remains
independent of the diagnostic counters.

This register stream is for **base M1 / T8103 only** and requires the native
Linux `ane` driver. It is not an implementation of CoreML or LUT6 compression.
The original advertised 32736 dimension bound is insufficient to establish
hardware support: an FP16 K=1024, N=32000 submission timed out. The new FP16
API rejects N>8192 before submission; our draft head uses 4096-row chunks
verified on hardware. See `notes/asahi/ane_fp16_probe.json`
in the repository root for tested shapes and errors.

`runq.c`, `ane/prefill.h`, and the model headers are retained as the source for
future native target verification. The current Linux DFlash experiment uses
the copied matrix backend for the draft and MLX/Vulkan for the target.
