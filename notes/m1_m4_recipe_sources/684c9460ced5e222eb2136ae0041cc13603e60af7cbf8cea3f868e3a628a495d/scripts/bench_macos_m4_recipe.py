"""Measure the M4 full-ANE offload method on M1 with the public 0.6B pair.

Four prompts, paired trials, fresh caches and exclusive ANE/GPU reservations.
Compare full-ANE SD with both the unchanged MLX bf16 Metal target and ordinary
greedy decoding through the same LUT6 ANE target. The latter comparison isolates
speculation from quantization and changing the target's execution engine.
All timings exclude model loading, compilation and prompt prefill.
"""
from __future__ import annotations
import macos_m4_environment
import argparse
import gc
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import bench_macos_dflash as bench
from macos_m4_coreml_runtime import FullANEStack, generate


def cosine(a,b):
    a=np.asarray(a,np.float32).reshape(-1);b=np.asarray(b,np.float32).reshape(-1)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def comparison(a,b):
    first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    if first is None and len(a)!=len(b): first=min(len(a),len(b))
    return {"identical":a==b,"first_mismatch":first,
            "equal_prefix":len(a) if first is None else first,
            "positions_equal":sum(x==y for x,y in zip(a,b)),"lengths":[len(a),len(b)]}


def validate(stack,model,tok,save):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from dflash_mlx_speculators import DFlash
    draft=DFlash(Path(stack.manifest["draft"]),7,"fp16",model.model.embed_tokens.weight)
    rows=[]
    for name,prompt in bench.PROMPTS:
        stack.reset();draft.reset();cache=make_prompt_cache(model)
        ids=tok.encode(prompt);feature_cos=[]
        last_preds=None
        for start in range(0,len(ids),stack.B):
            part=ids[start:start+stack.B]
            expected,features=bench.target_forward(model,part,cache,draft.feature_ids)
            actual,ane_features,pending=stack.forward(part)
            stack.commit(pending,len(part));stack.append_draft(ane_features)
            draft.append(mx.array(ane_features))
            reference=np.asarray(features.astype(mx.float32))
            feature_cos.append([cosine(ane_features[:,i*stack.W:(i+1)*stack.W],
                                       reference[:,i*stack.W:(i+1)*stack.W])
                                for i in range(len(draft.feature_ids))])
            last_preds={"ane":actual[-1],"mlx":expected[-1]}
        context_cos=[]
        for i,(k,v) in enumerate(draft.cache):
            context_cos.append({"k":cosine(stack.draft_k[i,:,:,:stack.offset],np.asarray(k.astype(mx.float32))),
                                "v":cosine(stack.draft_v[i,:,:,:stack.offset],np.asarray(v.astype(mx.float32)))})
        actual,hidden=stack.propose(last_preds["ane"])
        expected,reference=draft.propose(last_preds["ane"])
        row={"name":name,"prompt_tokens":len(ids),"target_feature_cosines":feature_cos,
             "target_next_token":last_preds,"context_cosines":context_cos,
             "draft_cosine":cosine(hidden,np.asarray(reference.astype(mx.float32))),
             "draft_ids":actual,"reference_draft_ids":expected,
             "draft_ids_agree":sum(x==y for x,y in zip(actual,expected)),
             "finite":bool(np.isfinite(hidden).all()),"nonzero":int(np.count_nonzero(hidden))}
        rows.append(row);save(rows)
        print("[validate] "+json.dumps({k:v for k,v in row.items() if k not in
              ("target_feature_cosines","context_cosines","draft_ids","reference_draft_ids")}),flush=True)
    del draft;gc.collect();mx.clear_cache()
    if not all(r["finite"] and r["nonzero"] and r["draft_cosine"]>.9 for r in rows):
        raise RuntimeError("ANE draft failed finite/nonzero/real-input cosine validation")
    return rows


def summarize(rows):
    summary={}
    for label in ("mlx_bf16","ane_lut6_ar","ane_lut6_sd"):
        group=[r for r in rows if r["impl"]==label]
        if group:
            summary[label]={"mean_tok_s":statistics.mean(r["tok_per_s_decode"] for r in group),
                            "trials":len(group),"per_prompt":{name:statistics.mean(
                            r["tok_per_s_decode"] for r in group if r["name"]==name)
                            for name,_ in bench.PROMPTS if any(r["name"]==name for r in group)}}
    pairs=[]
    for sd in (r for r in rows if r["impl"]=="ane_lut6_sd"):
        group={r["impl"]:r for r in rows if (r["name"],r["repeat"])==(sd["name"],sd["repeat"])}
        if len(group)!=3: continue
        pairs.append({"name":sd["name"],"repeat":sd["repeat"],
                      "vs_mlx":sd["tok_per_s_decode"]/group["mlx_bf16"]["tok_per_s_decode"],
                      "vs_ane_ar":sd["tok_per_s_decode"]/group["ane_lut6_ar"]["tok_per_s_decode"],
                      "tokens_vs_mlx":comparison(sd["tokens"],group["mlx_bf16"]["tokens"]),
                      "tokens_vs_ane_ar":comparison(sd["tokens"],group["ane_lut6_ar"]["tokens"])})
    summary["pairs"]=pairs
    if pairs:
        for label in ("vs_mlx","vs_ane_ar"):
            summary[label]={"mean_paired_speedup":statistics.mean(p[label] for p in pairs),
                            "max_trial_speedup":max(p[label] for p in pairs),
                            "per_prompt":{name:statistics.mean(p[label] for p in pairs if p["name"]==name)
                             for name,_ in bench.PROMPTS if any(p["name"]==name for p in pairs)}}
        for label in ("tokens_vs_mlx","tokens_vs_ane_ar"):
            summary[label]={"identical_trials":sum(p[label]["identical"] for p in pairs),
                            "trials":len(pairs),"positions_equal":sum(p[label]["positions_equal"] for p in pairs),
                            "total_positions":sum(p[label]["lengths"][0] for p in pairs)}
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts",type=Path,default=bench.REPO/".asahi/m4-recipe-m1")
    ap.add_argument("--max-new",type=int,default=100)
    ap.add_argument("--repeats",type=int,default=2)
    ap.add_argument("--out",type=Path,default=bench.REPO/"notes/m1_m4_recipe.json")
    ap.add_argument("--skip-validation",action="store_true")
    ap.add_argument("--native",action="store_true",help="Use the native Swift runner for timed ANE trials")
    args=ap.parse_args(argv)
    mx=bench.require_metal()
    from mlx_lm import load
    result={"started_utc":datetime.now(timezone.utc).isoformat(),"status":"waiting_for_devices",
            "machine":mx.device_info(),"macos":platform.mac_ver()[0],"procedure":__doc__,
            "max_new":args.max_new,"repeats":args.repeats,"rows":[],
            "ane_runtime":"Swift/CoreML" if args.native else "Python/CoreML",
            "source_sha256":{name:bench.sha256(bench.REPO/"scripts"/name) for name in
            ("bench_macos_m4_recipe.py","macos_m4_coreml_runtime.py","macos_m4_environment.py")},
            "packages":{p:importlib.metadata.version(p) for p in ("mlx","mlx-lm","coremltools")}}
    def save():
        result["summary"]=summarize(result["rows"])
        args.out.write_text(json.dumps(result,indent=2)+"\n")
    def save_validation(rows): result["validation"]=rows;save()
    save()
    native=None
    try:
        with bench.device_locks():
            result["status"]="loading";result["host_before"]=bench.host_state();save()
            start=time.perf_counter();stack=FullANEStack(args.artifacts)
            result["ane_load_s"]=time.perf_counter()-start
            result["manifest"]=stack.manifest
            start=time.perf_counter();model,tok=load(stack.manifest["target"])
            model.eval();result["mlx_load_s"]=time.perf_counter()-start
            if model.model.embed_tokens.weight.dtype!=mx.bfloat16: raise RuntimeError("Need unchanged bf16 MLX baseline")
            result["target_weights_sha256"]=bench.sha256(Path(stack.manifest["target"])/"model.safetensors")
            result["draft_weights_sha256"]=bench.sha256(Path(stack.manifest["draft"])/"model.safetensors")
            result["status"]="validating";save()
            if not args.skip_validation: validate(stack,model,tok,save_validation)
            if args.native:
                from macos_m4_native_runtime import NativeStack
                reference={name:generate(stack,tok,prompt,12,True) for name,prompt in bench.PROMPTS}
                manifest=stack.manifest
                del stack;gc.collect()
                native=NativeStack(args.artifacts,tok)
                result["native_load"]=native.ready
                result["source_sha256"]["m1_full_ane.swift"]=bench.sha256(bench.REPO/"swift-bench/m1_full_ane.swift")
                result["source_sha256"]["macos_m4_native_runtime.py"]=bench.sha256(bench.REPO/"scripts/macos_m4_native_runtime.py")
                result["native_binary_sha256"]=bench.sha256(args.artifacts/"m1-full-ane")
                result["native_parity"]=[]
                for name,prompt in bench.PROMPTS:
                    observed=native.generate(prompt,12,True)
                    check={"name":name,**comparison(observed["tokens"],reference[name]["tokens"]),
                           "native":observed,"python":reference[name]}
                    result["native_parity"].append(check);save()
                    print(f"[native parity] {name}: {check['identical']}",flush=True)
                if not all(r["identical"] for r in result["native_parity"]):
                    raise RuntimeError("Swift/Python CoreML token parity failed")
                def run_ane(prompt,limit,speculative): return native.generate(prompt,limit,speculative)
            else:
                def run_ane(prompt,limit,speculative): return generate(stack,tok,prompt,limit,speculative)
            result["status"]="warming";save()
            for speculative in (False,True): run_ane("Hello",10,speculative)
            bench.stock_baseline(model,tok,"Hello",10)
            counter=args.artifacts/"anebytes"
            if counter.exists():
                result["idle_counter"]=subprocess.check_output([str(counter),"2"],text=True,timeout=10).strip()
                proc=subprocess.Popen([str(counter),"2"],stdout=subprocess.PIPE,text=True)
                time.sleep(.15)
                active=run_ane(bench.PROMPTS[0][1],40,True)
                result["active_counter"]=proc.communicate(timeout=10)[0].strip()
                result["counter_probe"]={"tokens":len(active["tokens"]),"calls":active["calls"]};save()
            result["status"]="benchmarking";save()
            for repeat in range(args.repeats):
                for name,prompt in bench.PROMPTS:
                    order=("mlx_bf16","ane_lut6_ar","ane_lut6_sd") if repeat%2==0 else (
                           "ane_lut6_sd","ane_lut6_ar","mlx_bf16")
                    for label in order:
                        if label=="mlx_bf16":
                            row=bench.stock_baseline(model,tok,prompt,args.max_new)
                            row["tok_per_s_decode"]=row["generation_tps"]
                            row["text"]=tok.decode(row["tokens"])
                        else: row=run_ane(prompt,args.max_new,label=="ane_lut6_sd")
                        row.update(impl=label,name=name,repeat=repeat+1)
                        result["rows"].append(row);save()
                        print("[trial] "+json.dumps({"impl":label,"name":name,"repeat":repeat+1,
                              "tokens":len(row["tokens"]),"tok_s":row["tok_per_s_decode"],
                              "cycles":row.get("cycles"),"accepted":row.get("accepted"),
                              "profile_s":row.get("profile_s")}),flush=True)
            result["host_after"]=bench.host_state();result["status"]="complete";save()
            print("[summary] "+json.dumps(result["summary"]),flush=True)
    except BaseException as error:
        result["status"]="failed";result["error"]=repr(error);save();raise
    finally:
        if native is not None: native.close()
    return 0


if __name__=="__main__": raise SystemExit(main())
