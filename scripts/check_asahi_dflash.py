"""Compare ANE linears on real DFlash inputs against rounded FP32 CPU products.

This isolates native projection correctness, not agreement with every possible
framework attention kernel. It compares the final draft hidden vectors and
proposal IDs after multiple context updates, using the published trained draft.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from asahi_ane import ANE, REPO
from asahi_dflash import DFlash
from bench_asahi_dflash import device_locks, target_forward
from bench_asahi_mlx import TARGET, TARGET_REV, model_snapshot, sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, default=REPO / ".asahi/models/microcycle-dflash")
    ap.add_argument("--out", type=Path, default=REPO / "notes/asahi/dflash_projection_check.json")
    args = ap.parse_args()
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    with device_locks(), ANE() as ane:
        native = DFlash(args.draft, ane)
        reference = DFlash(args.draft, None)
        path, _ = model_snapshot(TARGET, TARGET_REV)
        model, tok = load(str(path))
        ids = tok.encode("The capital of France is Paris, which is known for")
        target_cache = make_prompt_cache(model)
        next_ids, features = target_forward(model, ids, target_cache, native.feature_ids)
        rows = []
        for count in (len(ids) - 1, 1):
            begin = native.offset
            native.append(features[begin:begin + count])
            reference.append(features[begin:begin + count])
            anchor = next_ids[native.offset - 1]
            n_ids, n_hidden = native.propose(anchor)
            r_ids, r_hidden = reference.propose(anchor)
            cosine = float(np.sum(n_hidden * r_hidden) /
                           np.sqrt(np.sum(n_hidden**2) * np.sum(r_hidden**2)))
            relative_l2 = float(np.linalg.norm(n_hidden - r_hidden) / np.linalg.norm(r_hidden))
            row = {"committed_context": native.offset, "cosine": cosine,
                   "relative_l2": relative_l2, "max_absolute_error": float(np.max(np.abs(n_hidden - r_hidden))),
                   "native_ids": n_ids, "reference_ids": r_ids,
                   "top1_agreement": sum(a == b for a, b in zip(n_ids, r_ids)) / len(n_ids),
                   "passed": cosine > 0.999 and relative_l2 < 0.02}
            rows.append(row)
            print(json.dumps(row), flush=True)
        result = {"scope": __doc__, "weights_sha256": sha256(args.draft / "model.safetensors"),
                  "native_library_sha256": sha256(REPO / ".asahi/build/libane_sd.so"),
                  "rows": rows, "submissions": ane.submissions,
                  "passed": all(r["passed"] for r in rows)}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
