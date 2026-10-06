"""Verify M1 Linux ANE replay, then benchmark the complete SD host loop.

plan is read-only and works on macOS. verify and bench submit programs only
on base M1 Asahi Linux with the expected ANE accel driver. A successful macOS
capture does not imply successful Linux execution or the same speedup.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time
import numpy as np
from bench_macos_dflash import device_locks
from capture_m1_full_stack import sha256,pack_tensor
from m1_lut6_weights import Checkpoint,prepare
from m1_ane_replay import Program,device_path,plan
from m1_full_ane_host import FullANEStack,generate


def check_sources(root,templates=True):
    stack=json.loads((root/'stack.json').read_text())
    plans={}
    for info in stack['artifacts']:
        p=root/'programs'/info['program']
        if templates:
            from package_m1_full_stack import verify_or_extract
            if not plans:verify_or_extract(root)
            recipe=json.loads((p/'packing.json').read_text())
            data=gzip.decompress((p/recipe['template']).read_bytes())
            if recipe['sha256']!=info['hwx_sha256']:raise ValueError('Wrong reconstructed program identity')
            for op in recipe['operations']:
                size=128+16*op['shape'][1]*6//8 if 'group_offsets' in op else op['size']//16 if 'engine_offsets' in op else op['size']
                for index,offset in enumerate(op.get('group_offsets',op.get('engine_offsets',[op.get('offset')]))):
                    extent=size
                    if offset is None or offset<0 or offset+extent>len(data) or any(data[offset:offset+extent]):
                        raise ValueError('Learned coefficients remain in kernel template')
            actual=plan(p,data);actual['hwx_sha256']=recipe['sha256']
        else:
            if sha256(p/'hwx/model.hwx')!=info['hwx_sha256']:raise ValueError('Changed reconstructed HWX')
            actual=plan(p)
        expected=json.loads((p/'linux-plan.json').read_text())
        if actual!=expected:raise ValueError(f'Changed replay plan: {info["program"]}')
        plans[info['program']]=actual
    if len(plans)!=5:raise ValueError('Expected two target chunks, draft, projector and shared full head')
    return stack,plans


def embedding(target,stack):
    if sha256(target/'model.safetensors')!=stack['target']['safetensors_sha256']:
        raise ValueError('Download the pinned BF16 checkpoint with hf download')
    reader=Checkpoint(target,stack['target']['safetensors_sha256'])
    try:values=reader.read('model.embed_tokens.weight')
    finally:reader.close()
    if hashlib.sha256(values.astype('<f2').tobytes()).hexdigest()!=stack['embedding_raw_sha256']:
        raise ValueError('Embedding reconstruction differs from measured macOS run')
    return values


def verify_tensors(values,expected):
    checks=[]
    for tensor in expected:
        value=values[tensor['name']]
        raw=pack_tensor(value,tensor['compiler_geometry'])
        digest=hashlib.sha256(raw).hexdigest()
        checks.append({'name':tensor['name'],'sha256':digest,'expected_sha256':tensor['sha256'],
                       'byte_identical':list(value.shape)==tensor['shape'] and digest==tensor['sha256']})
    if not all(check['byte_identical'] for check in checks):
        raise RuntimeError('Regenerated tensor differs from macOS golden: '+str(checks))
    return checks


def verify_cases(root,programs,capture,values,result,save):
    """Regenerate real prompt/cache inputs; retain only captured hashes in Git."""
    cases={(row['prompt'],row['label'],row['stage']):row for row in capture['cases']}
    state={}
    class Wrapper:
        def __init__(self,info):
            self.name=info['program']
            self.label=f"target{info['start']}" if info['role']=='target' else info['role']
        def predict(self,inputs,role=None):
            label=role or self.label
            offset=host.draft_offset if self.name=='projector' else host.offset
            captured=state['captured']
            stage='early' if (label,'early') not in captured else (
                'decode' if offset>=state['prompt_length'] and (label,'decode') not in captured else (
                'late' if offset>=80 and (label,'late') not in captured else None))
            row=cases[(state['prompt'],label,stage)] if stage else None
            if row:
                if row['offset']!=offset:raise RuntimeError('Regenerated cache offset differs')
                verify_tensors(inputs,row['inputs'])
            outputs=programs[self.name].predict(inputs)
            if row:
                checks=verify_tensors(outputs,row['outputs'])
                result['cases'].append({'case':row['directory'],'inputs_byte_identical':True,'checks':checks})
                captured.add((label,stage));save()
            return outputs
    host=FullANEStack(root,values,Wrapper)
    class ReceiptTokenizer:
        eos_token_ids={host.cfg['eos_token_id']}
        def encode(self,prompt):return next(g['prompt_tokens'] for g in capture['generations'] if g['prompt']==prompt)
        def decode(self,tokens):return ''
    for g in capture['generations']:
        state.update(prompt=g['name'],prompt_length=len(g['prompt_tokens']),captured=set())
        actual=generate(host,ReceiptTokenizer(),g['prompt'],100,True)
        if actual['tokens']!=g['tokens'] or actual['trace']!=g['trace']:
            raise RuntimeError('Regenerated SD trace differs from measured macOS reference')
    if len(result['cases'])!=len(capture['cases']):raise RuntimeError('Incomplete regenerated golden coverage')
    result['goldens_verified']=True;save()


class MeasuredProgram:
    def __init__(self,program,owner,label):self.program,self.owner,self.label=program,owner,label
    def predict(self,inputs,role=None):
        begin=time.perf_counter();outputs=self.program.predict(inputs)
        label=role or self.label
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-begin
        self.owner.calls[label]=self.owner.calls.get(label,0)+1
        return outputs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--mode',choices=('plan','reconstruct','verify','bench'),default='plan')
    ap.add_argument('--device');ap.add_argument('--target',type=Path)
    ap.add_argument('--draft',type=Path)
    ap.add_argument('--cache',type=Path,default=Path('.asahi/m1-full-ane-cache'))
    ap.add_argument('--max-new',type=int,default=100);ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.max_new<1 or args.repeats<1:ap.error('Need positive token limit and repeats')
    stack,plans=check_sources(args.root)
    if args.mode=='plan':
        print(json.dumps({'status':'planned','linux_hardware_verified':False,
                          'programs':{k:{key:p[key] for key in ('td_count','td_size','tsk_size','active_bars')} for k,p in plans.items()}},indent=2))
        return
    if not args.target or not args.draft:ap.error('reconstruct/verify/bench require pinned --target and --draft HF downloads')
    if args.mode=='reconstruct':
        runtime=prepare(args.root,args.target,args.draft,args.cache)
        _,actual=check_sources(runtime,templates=False)
        print(json.dumps({'status':'reconstructed','cache':str(runtime),'full_hwx_byte_identical':True,
                          'program_sha256':{name:p['hwx_sha256'] for name,p in actual.items()}},indent=2))
        return
    if not args.out or args.out.exists():ap.error('--out must name a fresh receipt')
    result={'status':'waiting','host':platform.platform(),'mode':args.mode,'cases':[],'rows':[],
            'max_new':args.max_new,'repeats':args.repeats,'timing':'decode only, last prompt token included; prefill and loading excluded',
            'source_sha256':{n:sha256(Path(__file__).parent/n) for n in
                             ('run_m1_full_ane_replay.py','m1_ane_replay.py','m1_full_ane_host.py','m1_lut6_weights.py','capture_m1_full_stack.py')},
            'model_checkpoints':{name:stack[name] for name in ('target','draft')},
            'program_sha256':{k:p['hwx_sha256'] for k,p in plans.items()},'goldens_verified':False}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    save();fd=None;programs={}
    try:
        with device_locks():
            path=device_path(args.device);result['device']=str(path)
            capture=json.loads((args.root/'cases/capture.json').read_text())
            result['status']='reconstructing';save()
            runtime=prepare(args.root,args.target,args.draft,args.cache)
            check_sources(runtime,templates=False)
            values=embedding(args.target,stack)
            fd=os.open(path,os.O_RDWR|os.O_CLOEXEC)
            for name in plans:
                example=next(r for r in capture['cases'] if r['program']==name)
                shapes={t['name']:t['shape'] for t in example['outputs']}
                programs[name]=Program(fd,runtime/'programs'/name,shapes)
            result['status']='verifying';save()
            verify_cases(args.root,programs,capture,values,result,save)
            if args.mode=='bench':
                import mlx.core as mx
                from mlx_lm import load
                from bench_macos_dflash import generate as baseline
                model,tok=load(str(args.target))
                # The factory is called during construction; bind owners afterwards.
                wrappers=[]
                def factory(info):
                    label=f"target{info['start']}" if info['role']=='target' else info['role']
                    p=MeasuredProgram(programs[info['program']],None,label);wrappers.append(p);return p
                host=FullANEStack(args.root,values,factory)
                for p in wrappers:p.owner=host
                result['status']='benchmarking';save()
                for repeat in range(args.repeats):
                    for g in capture['generations']:
                        rows={}
                        order=('mlx_bf16','ane_lut6_ar','ane_lut6_sd') if repeat%2==0 else ('ane_lut6_sd','ane_lut6_ar','mlx_bf16')
                        for label in order:
                            row=baseline(model,tok,None,g['prompt'],args.max_new,False) if label=='mlx_bf16' else generate(host,tok,g['prompt'],args.max_new,label=='ane_lut6_sd')
                            row.update(impl=label,name=g['name'],repeat=repeat+1)
                            result['rows'].append(row);rows[label]=row;save()
                            print(g['name'],repeat+1,label,round(row['tok_per_s_decode'],2),flush=True)
                        if rows['ane_lut6_ar']['tokens']!=rows['ane_lut6_sd']['tokens']:
                            raise RuntimeError('SD token identity failed against same LUT6 ANE target')
                        if args.max_new==100 and rows['ane_lut6_sd']['tokens']!=g['tokens']:
                            raise RuntimeError('Linux generation differs from captured macOS compressed target')
                means={label:statistics.mean(r['tok_per_s_decode'] for r in result['rows'] if r['impl']==label)
                       for label in ('mlx_bf16','ane_lut6_ar','ane_lut6_sd')}
                result['means']=means
                result['sd_vs_linux_mlx']=means['ane_lut6_sd']/means['mlx_bf16']
                result['sd_vs_padded_b8_ane_ar']=means['ane_lut6_sd']/means['ane_lut6_ar']
                result['per_prompt_means']={g['name']:{label:statistics.mean(r['tok_per_s_decode'] for r in result['rows']
                                                             if r['name']==g['name'] and r['impl']==label)
                                                        for label in means} for g in capture['generations']}
                paired=[]
                for sd in (r for r in result['rows'] if r['impl']=='ane_lut6_sd'):
                    ar=next(r for r in result['rows'] if r['impl']=='mlx_bf16' and r['name']==sd['name'] and r['repeat']==sd['repeat'])
                    paired.append(sd['tok_per_s_decode']/ar['tok_per_s_decode'])
                result['max_paired_speedup_vs_linux_mlx']=max(paired)
                result['target_quality']='LUT6 compressed target; BF16 token identity is not claimed'
            result['status']='complete';save()
    except BaseException as error:
        result['status']='failed';result['error']=str(error);save();raise
    finally:
        for program in programs.values():program.close()
        if fd is not None:os.close(fd)


if __name__=='__main__':main()
