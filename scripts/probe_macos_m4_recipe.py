"""Validate the first Core ML target layer on real inputs and ANE byte counters."""
import macos_m4_environment
import json
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
from transformers import AutoTokenizer
from asahi_dflash import Tensors
import bench_macos_dflash as bench
from convert_macos_m4_recipe import placement
from macos_m4_coreml_models import TargetChunk
from macos_m4_coreml_runtime import rope,causal_mask


def main():
    root=bench.REPO / ".asahi/m4-recipe-m1"
    compiled=root / "probe_target0_stable/lut6_per_grouped_channel.mlmodelc"
    target=bench.REPO / ".asahi/m1-sd-macos/target"
    result={"status":"waiting","rows":[]}
    output=bench.REPO / "notes/m1_m4_recipe_probe_runtime.json"
    def save(): output.write_text(json.dumps(result,indent=2)+"\n")
    save()
    with bench.device_locks():
        torch.set_num_threads(4)
        result["placement"]=placement(compiled)
        model=ct.models.CompiledMLModel(str(compiled),compute_units=ct.ComputeUnit.CPU_AND_NE)
        module=TargetChunk(target,0,1).eval()
        tok=AutoTokenizer.from_pretrained(str(target))
        reader=Tensors(target / "model.safetensors")
        for label,prompt in bench.PROMPTS:
            ids=tok.encode(prompt)[:8]
            hidden=np.zeros((1,8,1024),np.float16)
            hidden[0,:len(ids)]=reader.read("model.embed_tokens.weight",np.float16,ids)
            cos,sin=rope(0,8,128,1e6)
            inputs={"hidden":hidden,"cos":cos,"sin":sin,
                    "cache_k":np.zeros((1,1,8,256,128),np.float16),
                    "cache_v":np.zeros((1,1,8,256,128),np.float16),"mask":causal_mask(0,256,8)}
            with torch.no_grad(): expected=module(*(torch.from_numpy(x) for x in inputs.values()))[0].float().numpy()
            actual=np.asarray(model.predict(inputs)["hidden_out"],np.float32)
            row={"name":label,"finite":bool(np.isfinite(actual).all()),
                 "nonzero":int(np.count_nonzero(actual)),"max_abs":float(np.abs(actual).max()),
                 "cosine":float(np.sum(actual*expected)/(np.linalg.norm(actual)*np.linalg.norm(expected))),
                 "relative_rmse":float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))}
            result["rows"].append(row);print(json.dumps(row),flush=True);save()
        reader.close()
        samples=[]
        for _ in range(25):
            start=time.perf_counter();model.predict(inputs);samples.append((time.perf_counter()-start)*1000)
        result["predict_median_ms"]=statistics.median(samples[5:]);save()
        counter=root / "anebytes"
        if counter.exists():
            result["idle_counter"]=subprocess.check_output([str(counter),"2"],text=True,timeout=10).strip()
            proc=subprocess.Popen([str(counter),"2"],stdout=subprocess.PIPE,text=True)
            time.sleep(.2)
            stop=time.perf_counter()+1.5;calls=0
            while time.perf_counter()<stop: model.predict(inputs);calls+=1
            result["active_counter"]=proc.communicate(timeout=10)[0].strip()
            result["counter_predictions"]=calls
        result["status"]="complete"
        result["qualified"]=result["placement"]["math_all_ane"] and all(
            r["finite"] and r["nonzero"] and r["cosine"]>.99 for r in result["rows"])
        save();print(json.dumps({k:v for k,v in result.items() if k not in ("placement","rows")}),flush=True)


if __name__=="__main__": main()
