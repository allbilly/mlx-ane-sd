# macOS MLX and ANE reference survey

Checked 2026-10-06 on this M1 MacBook Air, 8 GB. Existing reference repositories and Asahi source files were inspected without changing them.

MLX's current official README lists CPU and GPU as its supported devices: [MLX](https://github.com/ml-explore/mlx#mlx). MLX does not dispatch a draft or target to ANE. A hybrid application must call a separate ANE runtime and coordinate its outputs with MLX's Metal computation. The earlier GPU-only SD sweep therefore did not test this repository's ANE claim.

| Local reference | Relevant implementation | Use in this experiment |
| --- | --- | --- |
| `~/Desktop/ANEForge` | Tensor graph → MIL → private e5rt/ANE dispatch. `aneforge/_runtime.py` distinguishes CPU mask `0x1`, GPU `0x2`, ANE `0x4`. Qwen attention/RoPE and vocabulary tiling are available. | A separate native draft runtime, with ANE-only mask `0x4`. Must check actual numerical outputs as well as successful compilation. |
| `~/Desktop/Orion` | Direct `_ANEInMemoryModel` compilation/evaluation and IOSurface inputs/outputs; avoids Core ML placement fallback. | The previously completed M1 capture kit already used this runtime for real ANE kernels. A lower-level alternative for the draft. |
| `~/more-ane-transformers` | Core ML transformer conversion, with an existing Python 3.11 environment containing Torch and coremltools. | Conversion reference and usable tooling if the direct runtime needs replacement. |
| `~/Desktop/anemll` | Core ML LLM conversion and chunked execution. | Reference for chunk/cache layouts; requires adapting the public DFlash checkpoint. |
| `~/ane-llm-measurements` | Core ML device placement and ANE counter measurements on M1. | Placement evidence: selecting `.cpuAndNeuralEngine` alone permits CPU fallback. |
| `~/Desktop/qwen3.c/macos` | Core ML Qwen3 target with KV state management, padded decode and operation placement reports. | Target-offload reference. Its dequantized weights and CPU head differ from the unchanged MLX bf16 baseline. |
| `~/ane` | Linux DRM/register execution and ANE reverse-engineering material. | Valuable hardware/dump reference; its Linux device backend is not the macOS runtime. |

Public upstream references: [ANEForge](https://github.com/sbryngelson/ANEForge), [Orion](https://github.com/allbilly/Orion), [Anemll](https://github.com/Anemll/Anemll), and Apple's [ANE transformer implementation](https://github.com/apple-aiml-research/ml-ane-transformers). Apple's implementation converts PyTorch models to Core ML; it does not add an ANE backend to MLX.

The runtime smoke test executed `x * 2` with ANE-only mask `0x4`, returning 2,047 nonzero values with zero maximum error. Successful execution alone was not enough for the draft: its real projector output reached 7,968 while ANEForge's `reduce_l2_norm`-based normalization returned all zeros. Intermediate-output checks localized the failure to normalization. The initial benchmark is explicitly invalid, even though serial target verification preserved output tokens. Rescaling before the reduction fixed the issue and passed real-input comparison with the MLX FP16 draft on all four prompts.

The corrected [native ANE experiment](m1_macos_ane_dflash.md) completed 24 trials with 1,946 native ANE evaluations. The body and full-vocabulary head execute on ANE; the target remains on MLX/Metal. Serial verification preserved all 800 token IDs but averaged 0.66× its paired stock baseline, with 1.02× best trial. These desktop-session measurements did not reproduce 3× and do not establish the ceiling for the README's larger full-ANE target stack.

This repository already separates the runtimes in `swift-bench/Sources/DFlashCore/DFlashANEDraft.swift`: the draft uses Core ML with `.cpuAndNeuralEngine`, while the target uses MLX. The Python Phase F.1 implementation also calls Core ML separately. The new small-model macOS experiment follows that separation with direct ANE dispatch rather than claiming MLX itself supports ANE.

The original README measurements use Qwen3-4B on an M4 Pro with 64 GB, including substantial target offload. A public Qwen3-0.6B DFlash + MLX target experiment on this 8 GB M1 tests the heterogeneous drafting mechanism, not an exact hardware/model reproduction of the full-ANE result. A correct draft and a four-prompt sweep are prerequisites for any speedup conclusion.
