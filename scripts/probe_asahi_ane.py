"""Check native FP16 ANE matrix shapes against CPU products and invalid inputs."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from asahi_ane import ANE, REPO
from bench_asahi_dflash import device_locks
from bench_asahi_mlx import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "notes/asahi/ane_fp16_probe.json")
    args = ap.parse_args()
    rng = np.random.default_rng(7)
    result = {"native_library_sha256": sha256(REPO / ".asahi/build/libane_sd.so"), "rows": []}
    with device_locks(), ANE() as ane:
        for k, n, rows in [(64, 96, 1), (128, 256, 8), (1024, 3072, 8),
                           (1024, 4096, 7), (1024, 8192, 7)]:
            w = (rng.normal(size=(n, k)) * .02).astype(np.float16)
            x = rng.normal(size=(rows, k)).astype(np.float32)
            op = ane.linear(w)
            begin = time.perf_counter()
            y = op(x)
            elapsed = time.perf_counter() - begin
            ref = x.astype(np.float16).astype(np.float32) @ w.astype(np.float32).T
            row = {"k": k, "n": n, "rows": rows, "elapsed_s": elapsed,
                   "max_error": float(np.max(np.abs(y - ref))),
                   "cosine": float(np.sum(y * ref) / np.sqrt(np.sum(y * y) * np.sum(ref * ref))),
                   "passed": bool(np.allclose(y, ref, atol=.01, rtol=.015))}
            result["rows"].append(row)
            print(json.dumps(row), flush=True)
            if not row["passed"]:
                raise RuntimeError("Native ANE matrix did not match CPU reference")
        before = ane.submissions
        for label, action in [("unvalidated_width", lambda: ane.linear(np.zeros((8193, 32), np.float16))),
                              ("nonfinite_input", lambda: op(np.full((1, k), np.nan, np.float32))),
                              ("nonfinite_weights", lambda: ane.linear(np.full((32, 32), np.nan, np.float16)))]:
            try:
                action()
            except RuntimeError:
                result[label + "_rejected"] = True
            else:
                raise RuntimeError(f"Invalid {label} was accepted")
        if ane.submissions != before:
            raise RuntimeError("An invalid input reached ANE submission")
        result["invalid_inputs_submitted"] = False
        result["submissions"] = ane.submissions
    result["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
