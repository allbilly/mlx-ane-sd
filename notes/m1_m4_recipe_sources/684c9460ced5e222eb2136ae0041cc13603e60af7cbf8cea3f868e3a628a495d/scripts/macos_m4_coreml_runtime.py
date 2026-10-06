"""Core ML cached draft + chunked full target, following the M4 offload recipe."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import coremltools as ct


def rope(offset, block, dim, theta):
    inv = theta**(-np.arange(dim//2,dtype=np.float32)/(dim//2))
    angles = np.arange(offset,offset+block,dtype=np.float32)[:,None]*inv
    angles = np.concatenate((angles,angles),axis=-1)
    return (np.cos(angles).astype(np.float16)[None,None],
            np.sin(angles).astype(np.float16)[None,None])


def causal_mask(offset, capacity, block):
    result=np.full((1,1,block,capacity+block),-10000,np.float16)
    result[:,:,:,:offset]=0
    result[:,:,:,capacity:]=np.where(np.tri(block,dtype=bool),0,-10000)
    return result


class CoreMLProgram:
    def __init__(self, info, owner, role):
        self.owner,self.role=owner,role
        if not info["placement"]["math_all_ane"]:
            raise RuntimeError(f"{role}: refusing unverified ANE math placement: {info['placement']['math_counts']}")
        self.model=ct.models.CompiledMLModel(info["compiled"],compute_units=ct.ComputeUnit.CPU_AND_NE)

    def predict(self, inputs, role=None):
        start=time.perf_counter()
        result=self.model.predict(inputs)
        label=role or self.role
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-start
        self.owner.calls[label]=self.owner.calls.get(label,0)+1
        return result


class FullANEStack:
    def __init__(self, directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory / "manifest.json").read_text())
        m=self.manifest
        if m["status"]!="complete" or not m["math_all_ane"]:
            raise RuntimeError("Need a completed manifest with verified ANE math placement")
        self.cfg,self.dc=m["config"],m["draft_config"]["transformer_layer_config"]
        self.B,self.C=m["block"],m["capacity"]
        self.D,self.KV=self.cfg["head_dim"],self.cfg["num_key_value_heads"]
        self.W=self.cfg["hidden_size"]
        self.feature_ids=tuple(m["draft_config"]["aux_hidden_state_layer_ids"])
        self.embedding=np.load(self.directory / "embedding.npy",mmap_mode="r")
        self.profile={};self.calls={}
        self.chunks=[]
        for info in m["artifacts"]:
            role=info["role"]
            program=CoreMLProgram(info,self,f"target{info['start']}" if role=="target" else role)
            if role=="target": self.chunks.append((info,program))
            elif role=="head": self.head=program
            elif role=="projector": self.projector=program
            elif role=="draft": self.draft=program
        self.reset()

    def reset(self):
        self.offset=self.draft_offset=0
        self.cache_k=np.zeros((self.cfg["num_hidden_layers"],1,self.KV,self.C,self.D),np.float16)
        self.cache_v=np.zeros_like(self.cache_k)
        self.draft_k=np.zeros((self.dc["num_hidden_layers"],1,self.KV,self.C,self.D),np.float16)
        self.draft_v=np.zeros_like(self.draft_k)
        self.profile={};self.calls={}

    def ids(self, hidden, start=0, end=None, role="target_head"):
        end=self.B if end is None else end
        outputs=self.head.predict({"hidden":hidden},role)
        begin=time.perf_counter()
        rows=end-start
        best=np.full(rows,-np.inf,np.float32)
        ids=np.zeros(rows,np.int32)
        for i in range((self.cfg["vocab_size"]+8191)//8192):
            # Only reduce rows actually requested. fp32 NumPy argmax is faster
            # than its fp16 path, and converting preserves every fp16 value.
            logits=np.asarray(outputs[f"logits{i}"])[0,start:end].astype(np.float32)
            index=logits.argmax(-1)
            values=logits[np.arange(rows),index]
            better=values>best
            ids[better]=index[better]+i*8192;best[better]=values[better]
        if not np.isfinite(best).all(): raise RuntimeError("Invalid ANE vocabulary logits")
        self.profile["host_argmax"]=self.profile.get("host_argmax",0)+time.perf_counter()-begin
        return ids.tolist()

    def forward(self, token_ids, logits=True):
        n=len(token_ids)
        if not 1<=n<=self.B or self.offset+n>self.C: raise ValueError("Invalid target block/context length")
        hidden=np.zeros((1,self.B,self.W),np.float16)
        hidden[0,:n]=self.embedding[token_ids]
        cos,sin=rope(self.offset,self.B,self.D,self.cfg["rope_theta"])
        mask=causal_mask(self.offset,self.C,self.B)
        pending=[];captures={}
        for info,model in self.chunks:
            start,count=info["start"],info["count"]
            out=model.predict({"hidden":hidden,"cos":cos,"sin":sin,"mask":mask,
                               "cache_k":self.cache_k[start:start+count],
                               "cache_v":self.cache_v[start:start+count]})
            hidden=np.asarray(out["hidden_out"],np.float16)
            if not np.isfinite(hidden[0,:n]).all(): raise RuntimeError("Non-finite target hidden state")
            pending.append((start,count,np.asarray(out["new_k"],np.float16),np.asarray(out["new_v"],np.float16)))
            if info["captures"]:
                for local,index in enumerate(info["captures"]): captures[index]=np.asarray(out["captures"])[local,0,:n]
        features=np.concatenate([captures[i] for i in self.feature_ids],axis=-1).astype(np.float16)
        self.last_hidden=hidden
        predictions=self.ids(hidden,end=n) if logits else None
        return predictions,features,pending

    def commit(self, pending, count):
        for start,layers,k,v in pending:
            self.cache_k[start:start+layers,:,:,self.offset:self.offset+count]=k[:,:,:,:count]
            self.cache_v[start:start+layers,:,:,self.offset:self.offset+count]=v[:,:,:,:count]
        self.offset+=count

    def append_draft(self, features):
        for begin in range(0,len(features),self.B):
            active=features[begin:begin+self.B]
            n=len(active)
            if self.draft_offset+n>self.C: raise ValueError("Draft cache capacity exceeded")
            padded=np.zeros((1,self.B,len(self.feature_ids)*self.W),np.float16)
            padded[0,:n]=active
            cos,sin=rope(self.draft_offset,self.B,self.D,self.cfg["rope_theta"])
            outputs=self.projector.predict({"features":padded,"cos":cos,"sin":sin})
            k,v=np.asarray(outputs["new_k"],np.float16),np.asarray(outputs["new_v"],np.float16)
            if not np.isfinite(k[:,:,:,:n]).all() or not np.isfinite(v[:,:,:,:n]).all():
                raise RuntimeError("Invalid draft context projection")
            self.draft_k[:,:,:,self.draft_offset:self.draft_offset+n]=k[:,:,:,:n]
            self.draft_v[:,:,:,self.draft_offset:self.draft_offset+n]=v[:,:,:,:n]
            self.draft_offset+=n

    def propose(self, anchor):
        if self.offset!=self.draft_offset: raise RuntimeError("Draft/target cache offset mismatch")
        mask_id=self.manifest["draft_config"]["mask_token_id"]
        noise=self.embedding[[anchor]+[mask_id]*(self.B-1)][None]
        cos,sin=rope(self.draft_offset,self.B,self.D,self.cfg["rope_theta"])
        out=self.draft.predict({"hidden":noise,"cos":cos,"sin":sin,
                               "cache_k":self.draft_k,"cache_v":self.draft_v,
                               "mask":causal_mask(self.draft_offset,self.C,self.B)})
        hidden=np.asarray(out["hidden_out"],np.float16)
        if not np.isfinite(hidden).all() or not np.any(hidden): raise RuntimeError("Invalid ANE draft hidden state")
        return self.ids(hidden,start=1,role="draft_head"),hidden[0,1:]


def generate(stack, tok, prompt, limit, speculative):
    stack.reset()
    prompt_ids=tok.encode(prompt)
    if len(prompt_ids)+limit>stack.C: raise ValueError("Prompt/generation exceed compiled context capacity")
    start=time.perf_counter()
    for begin in range(0,len(prompt_ids)-1,stack.B):
        ids=prompt_ids[begin:min(begin+stack.B,len(prompt_ids)-1)]
        _,features,pending=stack.forward(ids,False)
        stack.commit(pending,len(ids))
        if speculative: stack.append_draft(features)
    prefill=time.perf_counter()-start
    stack.profile={};stack.calls={}
    start=time.perf_counter()
    first,features,pending=stack.forward(prompt_ids[-1:])
    stack.commit(pending,1)
    if speculative: stack.append_draft(features)
    tokens=[first[0]];traces=[];accepted=proposed=0
    eos=tok.eos_token_ids if hasattr(tok,"eos_token_ids") else {tok.eos_token_id}
    while len(tokens)<limit and tokens[-1] not in eos:
        candidates=stack.propose(tokens[-1])[0] if speculative and limit-len(tokens)>1 else []
        candidates=candidates[:max(0,limit-len(tokens)-1)]
        predictions,features,pending=stack.forward([tokens[-1],*candidates])
        count=0
        for candidate,prediction in zip(candidates,predictions):
            if candidate!=prediction: break
            count+=1
        committed=[*candidates[:count],predictions[count]]
        for i,token in enumerate(committed):
            if token in eos: committed=committed[:i+1];break
        stack.commit(pending,len(committed))
        if speculative: stack.append_draft(features[:len(committed)])
        tokens.extend(committed)
        traces.append({"candidates":candidates,"predictions":predictions,"accepted":count,
                       "committed":committed,"offset":stack.offset})
        accepted+=min(count,len(committed));proposed+=len(candidates)
    elapsed=time.perf_counter()-start
    return {"tokens":tokens,"text":tok.decode(tokens),"decode_s":elapsed,"prefill_s":prefill,
            "tok_per_s_decode":len(tokens)/elapsed,"cycles":len(traces),"accepted":accepted,
            "proposed":proposed,"profile_s":dict(stack.profile),"calls":dict(stack.calls),"trace":traces}
