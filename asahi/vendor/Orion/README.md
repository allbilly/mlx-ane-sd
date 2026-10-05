# Orion capture components

These MIT-licensed files were copied from `allbilly/Orion` at the revision in
`UPSTREAM.json`. Its hashes describe the original upstream files.

The local runtime adds `orion_program_copy_artifacts` to retain the evaluated
program's temporary directory before cleanup. The remaining runtime, tensor
I/O and MIL builder code is unchanged. The capture runner uses this retained
directory as evidence of what it evaluated. Its separate offline HWX export
must be compared with those artifacts before treating it as that same program.

This code requires Apple Silicon macOS and private Apple frameworks. It does
not implement the Linux driver or import Apple's compiler into Linux.
