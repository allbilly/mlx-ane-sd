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
from bench_macos_dflash import device_locks,stock_baseline
from capture_m1_full_stack import sha256,pack_tensor
from m1_lut6_weights import Checkpoint,prepare
from m1_ane_replay import Program,CacheOutput,device_path,plan
from m1_full_ane_host import FullANEStack,generate,trace_matches


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
        def reset_transport(self):programs[self.name].reset_transport()
        def commit_cache(self,*args):programs[self.name].commit_cache(*args)
        def predict_token_ids(self,inputs,start,end,role=None):
            # Always verify complete physical outputs against captured hashes.
            # Compare the optimized reduction on the same submission afterwards.
            outputs=self.predict(inputs,role)
            program=programs[self.name]
            expected=program.read_token_ids(start,end,outputs=outputs)
            actual=program.read_token_ids(start,end)
            if actual!=expected:raise RuntimeError('Mapped vocabulary reduction differs from full readback')
            result['head_id_checks']=result.get('head_id_checks',0)+1
            return actual
        def predict_verified_ids(self,inputs,candidates,start,end,role=None):
            outputs=self.predict(inputs,role)
            program=programs[self.name]
            expected=program.read_token_ids(start,end,outputs=outputs)
            actual=program.read_verified_ids(candidates,start,end)
            count=0
            for candidate,prediction in zip(candidates,expected):
                if candidate!=prediction:break
                count+=1
            wanted=expected if program.verify_readback=='all' else expected[:count+1]
            if actual!=wanted:raise RuntimeError('Early-stop verification differs from full head readback')
            result['head_prefix_checks']=result.get('head_prefix_checks',0)+1
            # Keep complete historic traces during verification; benchmark traces
            # record only the predictions that affect emitted tokens/cache state.
            return expected
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
            program=programs[self.name]
            program.submit(inputs)
            outputs=program.read_outputs(full=True)
            if row:
                # Check the actual resident surfaces, not just the CPU mirror.
                program=programs[self.name]
                bindings={t['name']:t for t in program.meta['io'] if t['direction']=='inputs'}
                for tensor in row['inputs']:
                    binding=bindings[tensor['name']]
                    raw=program.buffers[binding['bank']].map[:binding['allocation_bytes']]
                    if hashlib.sha256(raw).hexdigest()!=tensor['sha256']:
                        raise RuntimeError('Resident input surface differs from macOS golden: '+tensor['name'])
                checks=verify_tensors(outputs,row['outputs'])
                result['cases'].append({'case':row['directory'],'inputs_byte_identical':True,
                                        'resident_inputs_byte_identical':True,'checks':checks})
                captured.add((label,stage));save()
            if program.kv_readback=='prefix':
                for tensor in program.meta['io']:
                    if tensor['direction']=='outputs' and tensor['name'] in ('new_k','new_v'):
                        value=CacheOutput(program,tensor)
                        if row:
                            for count in (1,4,8):
                                actual=value.read_prefix(count)
                                if actual.tobytes()!=outputs[tensor['name']][:,:,:,:count].tobytes():
                                    raise RuntimeError('Partial K/V readback differs from full output')
                                result['kv_prefix_checks']=result.get('kv_prefix_checks',0)+1
                        outputs[tensor['name']]=value
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
    def reset_transport(self):self.program.reset_transport()
    def commit_cache(self,*args):
        begin=time.perf_counter()
        self.program.commit_cache(*args)
        label=self.label+'_cache_update'
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-begin
    def predict(self,inputs,role=None):
        begin=time.perf_counter();outputs=self.program.predict(inputs)
        label=role or self.label
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-begin
        self.owner.calls[label]=self.owner.calls.get(label,0)+1
        return outputs
    def predict_token_ids(self,inputs,start,end,role=None):
        begin=time.perf_counter();ids=self.program.predict_token_ids(inputs,start,end)
        label=role or self.label
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-begin
        self.owner.calls[label]=self.owner.calls.get(label,0)+1
        return ids
    def __getattr__(self,name):
        if name=='predict_verified_ids' and self.program.verify_readback=='accepted-prefix':return self.verified
        raise AttributeError(name)
    def verified(self,inputs,candidates,start,end,role=None):
        begin=time.perf_counter();ids=self.program.predict_verified_ids(inputs,candidates,start,end)
        label=role or self.label
        self.owner.profile[label]=self.owner.profile.get(label,0)+time.perf_counter()-begin
        self.owner.calls[label]=self.owner.calls.get(label,0)+1
        return ids


def mlx_baseline(model,tokenizer,prompt,limit):
    """Use the same stock greedy generator and rate as the macOS receipt."""
    row=stock_baseline(model,tokenizer,prompt,limit)
    row['tok_per_s_decode']=row['generation_tps']
    row['text']=tokenizer.decode(row['tokens'])
    return row

def synchronize_mlx():
    import mlx.core as mx
    mx.synchronize()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--mode',choices=('plan','reconstruct','verify','bench'),default='plan')
    ap.add_argument('--device');ap.add_argument('--target',type=Path)
    ap.add_argument('--draft',type=Path)
    ap.add_argument('--cache',type=Path,default=Path('.asahi/m1-full-ane-cache'))
    ap.add_argument('--transport',choices=('reference','packed','resident'),default='resident')
    ap.add_argument('--head-readback',choices=('full','rows','mapped','native'),default='mapped')
    ap.add_argument('--verify-readback',choices=('all','accepted-prefix'),default='all')
    ap.add_argument('--kv-readback',choices=('full','prefix'),default='full')
    ap.add_argument('--native-threads',type=int,default=1)
    ap.add_argument('--cpu-affinity',help='Comma-separated Linux CPU IDs; applies to all three benchmark configurations')
    ap.add_argument('--cpu-util-min',type=int,help='Optional per-process Linux utilization clamp, 0..1024; applies to all configurations')
    ap.add_argument('--max-new',type=int,default=100);ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--warmup-new',type=int,default=100)
    ap.add_argument('--num-draft',type=int,help='Useful draft rows; physical ANE block remains unchanged')
    ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    cpu_hint=None
    if args.cpu_util_min is not None:
        from m1_scheduler import utilization_minimum
        cpu_hint=utilization_minimum(args.cpu_util_min)
    if not 1<=args.native_threads<=32:ap.error('--native-threads must be in 1..32')
    if args.cpu_affinity:
        affinity={int(cpu) for cpu in args.cpu_affinity.split(',')}
        if not affinity or not affinity<=os.sched_getaffinity(0):ap.error('--cpu-affinity must select available CPUs')
        os.sched_setaffinity(0,affinity)
    if args.max_new<1 or args.repeats<1 or args.warmup_new<1:ap.error('Need positive token limit, warmup and repeats')
    stack,plans=check_sources(args.root)
    if args.num_draft is None:args.num_draft=stack['block']-1
    if not 1<=args.num_draft<stack['block']:ap.error('--num-draft must be in 1..block-1')
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
    result={'status':'waiting','host':platform.platform(),'mode':args.mode,'transport':args.transport,
            'head_readback':args.head_readback,'verify_readback':args.verify_readback,
            'kv_readback':args.kv_readback,'cases':[],'rows':[],
            'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'cpu_utilization_hint':cpu_hint,
            'mlx_synchronized_between_trials':True,
            'num_draft':args.num_draft,
            'max_new':args.max_new,'repeats':args.repeats,'timing':'decode only; prefill, loading and warm-up excluded',
            'timing_by_impl':{'mlx_bf16':'stock stream_generate generation_tps; time to first token excluded',
                              'ane_lut6_ar':'host decode timer includes last prompt-token forward',
                              'ane_lut6_sd':'host decode timer includes last prompt-token forward'},
            'mlx_baseline':'mlx_lm.stream_generate, greedy temp=0, prefill_step_size=32; same as macOS receipt',
            'source_sha256':{n:sha256(Path(__file__).parent/n) for n in
                             ('run_m1_full_ane_replay.py','m1_ane_replay.py','m1_full_ane_host.py','m1_lut6_weights.py','capture_m1_full_stack.py','bench_macos_dflash.py')},
            'model_checkpoints':{name:stack[name] for name in ('target','draft')},
            'program_sha256':{k:p['hwx_sha256'] for k,p in plans.items()},'goldens_verified':False}
    if args.head_readback=='native':
        from m1_native_transport import library,configure
        configure(args.native_threads)
        result['native_transport']=library()[1]
        result['native_threads']=args.native_threads
        result['source_sha256']['m1_native_transport.py']=sha256(Path(__file__).parent/'m1_native_transport.py')
    from m1_scheduler import host_observation
    result['source_sha256']['m1_scheduler.py']=sha256(Path(__file__).parent/'m1_scheduler.py')
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
                programs[name]=Program(fd,runtime/'programs'/name,shapes,transport=args.transport,
                                       head_readback=args.head_readback,verify_readback=args.verify_readback,
                                       kv_readback=args.kv_readback)
            result['status']='verifying';save()
            verify_cases(args.root,programs,capture,values,result,save)
            if args.mode=='bench':
                from mlx_lm import load
                model,tok=load(str(args.target))
                model.eval()
                # The factory is called during construction; bind owners afterwards.
                wrappers=[]
                def factory(info):
                    label=f"target{info['start']}" if info['role']=='target' else info['role']
                    p=MeasuredProgram(programs[info['program']],None,label);wrappers.append(p);return p
                host=FullANEStack(args.root,values,factory)
                for p in wrappers:p.owner=host
                def trial(label,prompt,limit):
                    before=host_observation()
                    if label=='mlx_bf16':
                        row=mlx_baseline(model,tok,prompt,limit)
                        synchronize_mlx()
                    else:row=generate(host,tok,prompt,limit,label=='ane_lut6_sd',num_draft=args.num_draft)
                    row['system_before'],row['system_after']=before,host_observation()
                    return row
                result['status']='warming'
                result['warmup']={'max_new':min(args.warmup_new,args.max_new),'excluded_from_rows':True,'calls':[]};save()
                for g in capture['generations']:
                    for label in ('mlx_bf16','ane_lut6_ar','ane_lut6_sd'):
                        row=trial(label,g['prompt'],result['warmup']['max_new'])
                        result['warmup']['calls'].append({'name':g['name'],'impl':label,'tokens':len(row['tokens'])});save()
                result['status']='benchmarking';save()
                for repeat in range(args.repeats):
                    for g in capture['generations']:
                        rows={}
                        order=('mlx_bf16','ane_lut6_ar','ane_lut6_sd') if repeat%2==0 else ('ane_lut6_sd','ane_lut6_ar','mlx_bf16')
                        for label in order:
                            row=trial(label,g['prompt'],args.max_new)
                            row.update(impl=label,name=g['name'],repeat=repeat+1)
                            result['rows'].append(row);rows[label]=row;save()
                            print(g['name'],repeat+1,label,round(row['tok_per_s_decode'],2),flush=True)
                        if rows['ane_lut6_ar']['tokens']!=rows['ane_lut6_sd']['tokens']:
                            raise RuntimeError('SD token identity failed against same LUT6 ANE target')
                        if args.max_new==100 and rows['ane_lut6_sd']['tokens']!=g['tokens']:
                            raise RuntimeError('Linux generation differs from captured macOS compressed target')
                        if args.max_new==100 and args.num_draft==stack['block']-1 and not trace_matches(rows['ane_lut6_sd']['trace'],g['trace']):
                            raise RuntimeError('Linux verification decisions/cache state differ from macOS trace')
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
