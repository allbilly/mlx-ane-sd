"""Compare the native ANE draft with the MLX FP16 draft on real target features."""
from pathlib import Path
import json
import time
import numpy as np
import bench_macos_dflash as bench


def main():
    mx = bench.require_metal()
    result = {"status": "waiting", "rows": []}
    out = bench.REPO / "notes/m1_macos_ane_dflash_parity.json"
    def save():
        out.write_text(json.dumps(result, indent=2) + "\n")
    save()
    with bench.device_locks():
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from dflash_macos_ane import DFlashANE, setup_aneforge
        from dflash_mlx_speculators import DFlash
        af, provenance = setup_aneforge(Path.home() / "Desktop/ANEForge")
        result["runtime"] = provenance
        # First establish whether this runtime can produce nonzero, correct data.
        x = af.input((32, 64))
        smoke = af.compile(x * 2.0, opt=0,
                           build_dir=bench.REPO / ".asahi/macos-ane-sd/smoke")
        data = np.arange(32 * 64, dtype=np.float16).reshape(32, 64) / 1024
        observed = smoke(data)
        result["smoke"] = {"mask": smoke._prog._device_mask,
                           "max_error": float(np.max(np.abs(observed - data * 2))),
                           "nonzero": int(np.count_nonzero(observed))}
        smoke.release(); save()
        print("smoke", result["smoke"], flush=True)
        model, tok = load(str(bench.REPO / ".asahi/m1-sd-macos/target"))
        ane = DFlashANE(bench.REPO / ".asahi/models/microcycle-dflash",
                       Path.home() / "Desktop/ANEForge", head="gpu")
        reference = DFlash(bench.REPO / ".asahi/models/microcycle-dflash", precision="fp16")
        result["input_ports"] = {k: {"port": ane.body_ports[k], "shape": list(t.shape),
                                     "index": t.attrs.get("idx")}
                                 for k, t in ane.body_inputs.items()}
        for label, prompt in bench.PROMPTS:
            cache = make_prompt_cache(model)
            ids, features = bench.target_forward(model, tok.encode(prompt), cache, ane.feature_ids)
            ane.reset(); reference.reset()
            ane.append(features); reference.append(features)
            actual_ids, actual = ane.propose(ids[-1])
            expected_ids, expected = reference.propose(ids[-1])
            expected = np.array(expected).astype(np.float32)
            actual = actual.astype(np.float32)
            row = {"name": label, "anchor": ids[-1], "native_ids": actual_ids,
                   "mlx_fp16_ids": expected_ids, "native_text": tok.decode(actual_ids),
                   "mlx_text": tok.decode(expected_ids), "native_nonzero": int(np.count_nonzero(actual)),
                   "native_abs_max": float(np.abs(actual).max()),
                   "expected_abs_max": float(np.abs(expected).max()),
                   "rmse": float(np.sqrt(np.mean((actual - expected)**2))),
                   "cosine": float(np.sum(actual * expected) /
                       max(1e-30, float(np.linalg.norm(actual) * np.linalg.norm(expected))))}
            result["rows"].append(row); save()
            print(json.dumps(row), flush=True)
        ane.close()
        result["status"] = "complete"; save()


if __name__ == "__main__":
    main()
