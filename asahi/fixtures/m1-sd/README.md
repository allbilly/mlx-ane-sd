# Pinned real inputs for the M1 capture task

These four NPZ files contain intermediate tensors from the capital, Fibonacci,
math and story benchmark prompts, recorded by `scripts/profile_asahi_dflash.py`
on the base M1 under Asahi. They contain no model coefficient arrays. Keeping
them in Git lets the same M1 prepare the capture kit after booting macOS and
pulling this repository, without transferring an ignored Linux directory.

`SOURCE.json` records the model revisions, coefficient digest, software/source
provenance, profiling receipt and SHA256 of each NPZ. Each fixture contains the
first draft layer's norm/FFN inputs, real head input, and Q/K/V after Q/K norm,
RoPE and GQA expansion, plus committed-context length and anchor token. Use
`numpy.load(..., allow_pickle=False)`.

The preparer verifies fixture and draft weight hashes. Its default uses these
pinned fixtures. `--fixtures .asahi/macos-capture-inputs` selects fresh inputs
from a later Linux profiling run. Host references generated from these inputs
are numerical diagnostics; macOS ANE output becomes the replay oracle only
after execution.

See [task.md](../../../task.md) for preparation, capture and return steps.
