"""Find the first ANE numerical divergence in the real DFlash graph."""
import json
from pathlib import Path
import numpy as np
import bench_macos_dflash as bench


def main():
    mx = bench.require_metal()
    with bench.device_locks():
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache
        from dflash_macos_ane import DFlashANE
        model, tok = load(str(bench.REPO / ".asahi/m1-sd-macos/target"))
        ane = DFlashANE(bench.REPO / ".asahi/models/microcycle-dflash",
                       Path.home() / "Desktop/ANEForge", head="gpu")
        ids, features = bench.target_forward(model, tok.encode(bench.PROMPTS[0][1]),
                                             make_prompt_cache(model), ane.feature_ids)
        ane.append(features); ane.propose(ids[-1])
        from aneforge._compile import _topo, compile_multi
        order = _topo(ane.graph_out)
        selected = [t for t in order if t.op == "rms_norm"]
        # All normalizations plus their raw inputs distinguish overflow from a broken projection.
        selected = [v for t in selected for v in (t.srcs[0], t)]
        debug = compile_multi(selected, build_dir=bench.REPO / ".asahi/macos-ane-sd/stages")
        feed = {t._name: ane.last_feed[k] for k, t in ane.body_inputs.items()}
        observed = debug.prog.eval(feed)
        values = {}
        inputs = {id(t): mx.array(ane.last_feed[k]) for k, t in ane.body_inputs.items()}
        for t in order:
            s = [values[id(v)] for v in t.srcs]; a = t.attrs
            if t.op == "input": v = inputs[id(t)]
            elif t.op == "matmul": v = s[0] @ mx.array(a["wt"]).T if "wt" in a else s[0] @ s[1]
            elif t.op == "bmm": v = s[0] @ s[1]
            elif t.op == "muls": v = s[0] * a["k"]
            elif t.op == "adds": v = s[0] + a["k"]
            elif t.op == "rms_norm": v = mx.fast.rms_norm(s[0], mx.array(a["gamma"]), a["eps"])
            elif t.op == "reshape": v = s[0].reshape(t.shape)
            elif t.op == "transpose": v = s[0].transpose(a["perm"])
            elif t.op == "slice_by_size": v = s[0][tuple(slice(b, b+n) for b,n in zip(a["begin"],a["size"]))]
            elif t.op == "concat": v = mx.concatenate(s, axis=a["axis"])
            elif t.op == "add": v = s[0] + (s[1] if len(s)>1 else a["scalar"])
            elif t.op == "mul": v = s[0] * (s[1] if len(s)>1 else a["scalar"])
            elif t.op == "softmax": v = mx.softmax(s[0],axis=a["axis"])
            elif t.op == "silu": v = s[0] * mx.sigmoid(s[0])
            else: raise RuntimeError((t.op, a))
            values[id(t)] = v
        mx.eval(values[id(ane.graph_out)])
        rows=[]
        for t in selected:
            actual = observed[t._name].astype(np.float32)
            expected = np.array(values[id(t)]).astype(np.float32)
            row={"node":t._name,"op":t.op,"shape":list(t.shape),
                 "native_max":float(np.abs(actual).max()), "mlx_max":float(np.abs(expected).max()),
                 "nonzero":int(np.count_nonzero(actual)), "finite":bool(np.isfinite(actual).all()),
                 "rmse":float(np.sqrt(np.mean((actual-expected)**2)))}
            rows.append(row); print(json.dumps(row),flush=True)
        (bench.REPO / "notes/m1_macos_ane_stages.json").write_text(json.dumps(rows,indent=2)+"\n")
        debug.release(); ane.close()


if __name__ == "__main__":
    main()
