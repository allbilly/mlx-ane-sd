"""Capture real full-stack tensors and evaluate the exported M1 HWX itself.

This is correctness capture, not a timed speed benchmark. Use the exact
artifacts from the measured run; re-palettizing would change the target.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import plistlib
import subprocess
from pathlib import Path
import numpy as np


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(1024*1024),b''): h.update(data)
    return h.hexdigest()


def geometry(desc):
    if desc['Type']!='Float16' or desc['Interleave']!=1:
        raise ValueError('Only compiler-declared non-interleaved fp16 tensors are supported')
    shape=tuple(desc[k] for k in ('Batches','Channels','Depth','Height','Width'))
    strides=tuple(desc[k] for k in ('BatchStride','PlaneStride','DepthStride','RowStride'))+(2,)
    size=desc['BatchStride']*desc['Batches']
    extent=sum((n-1)*s for n,s in zip(shape,strides))+2
    if min(shape)<=0 or extent>size: raise ValueError('Invalid compiler tensor extent')
    return shape,strides,size


def pack_tensor(values,desc):
    shape,strides,size=geometry(desc)
    values=np.asarray(values,dtype='<f2')
    if values.size!=np.prod(shape) or not np.isfinite(values).all():
        raise ValueError('Wrong-size or nonfinite real tensor')
    raw=bytearray(size)
    np.ndarray(shape,dtype='<f2',buffer=raw,strides=strides)[...]=values.reshape(shape)
    return bytes(raw)


def unpack_tensor(raw,desc,logical_shape):
    shape,strides,size=geometry(desc)
    if len(raw)!=size: raise ValueError('Wrong-size surface output')
    return np.ndarray(shape,dtype='<f2',buffer=raw,strides=strides).copy().reshape(logical_shape)


def export_programs(args):
    from bench_macos_dflash import device_locks
    from m1_hwx_decode import decode
    if args.programs.exists(): raise ValueError('Use a fresh programs directory')
    manifest=json.loads((args.artifacts/'manifest.json').read_text())
    with device_locks():
        for info in manifest['artifacts']:
            name=f"target_{info['start']}" if info['role']=='target' else info['role']
            root=args.programs/name;root.mkdir(parents=True)
            source=Path(info['compiled'])
            (root/'original.mil').write_bytes((source/'model.mil').read_bytes())
            (root/'model.mil').write_text((source/'model.mil').read_text().replace('@model_path',str(source.resolve())))
            (root/'weights').symlink_to(source.resolve()/'weights',target_is_directory=True)
            subprocess.run([str(args.tool.resolve()),'export',str(root.resolve()),str((root/'hwx').resolve())],check=True)
            status=plistlib.loads((root/'hwx/model.hwx.status.plist').read_bytes())
            (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            decode(root/'hwx/model.hwx',root/'hwx/model.hwx-decoded')
            print('exported',name,flush=True)


def validate_mil(args):
    from bench_macos_dflash import device_locks
    from m1_ane_replay import plan
    path=args.out/'capture.json';result=json.loads(path.read_text())
    if len(result['generations'])!=4 or not all(g['matches_native_benchmark'] for g in result['generations']):
        raise ValueError('Need all four real generations matching the measured native SD run')
    def save(): path.write_text(json.dumps(result,indent=2)+'\n')
    with device_locks():
        for name in result['programs']:
            root=args.programs/name
            if sha256(root/'hwx/model.hwx')!=result['programs'][name]['hwx_sha256']:
                raise ValueError('HWX changed since tensor capture')
            mapping=plan(root)
            (root/'linux-plan.json').write_text(json.dumps(mapping,indent=2)+'\n')
            rows=[r for r in result['cases'] if r['program']==name]
            batch=[{'case':str((args.out/r['directory']/'case.json').resolve()),
                    'output':str((args.out/r['directory']/'mil-eval-batch').resolve())} for r in rows]
            if any(Path(r['output']).exists() for r in batch): raise ValueError('Refusing to overwrite prior MIL evaluations')
            batch_file=root/'mil-batch.json';batch_file.write_text(json.dumps(batch,indent=2)+'\n')
            proc=subprocess.run([str(args.tool.resolve()),'eval-mil-batch',str(root.resolve()),str(batch_file.resolve())],
                                capture_output=True,text=True,timeout=240)
            (root/'mil-eval.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode: raise RuntimeError(proc.stderr[-2000:])
            for row in rows:
                case=args.out/row['directory'];checks=[]
                for t in row['outputs']:
                    actual_path=case/'mil-eval-batch'/f"output-{t['index']}.bin"
                    actual=unpack_tensor(actual_path.read_bytes(),t['compiler_geometry'],t['shape'])
                    checks.append({'name':t['name'],'byte_identical':sha256(actual_path)==t['sha256'],
                                   'sha256':sha256(actual_path),'finite':bool(np.isfinite(actual).all())})
                row['mil_evaluated_on_ane']=True;row['mil_output_checks']=checks;save()
                if not all(c['byte_identical'] and c['finite'] for c in checks):
                    raise RuntimeError('Standalone MIL differs from CoreML golden')
            print('verified MIL',name,len(rows),'all outputs byte-identical',flush=True)
    result['status']='complete_coreml_and_mil_goldens'
    result['offline_hwx_evaluated']=False
    result['hwx_runtime']='Compiler export; direct precompiled loading and Linux replay are not established by MIL validation'
    save()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--programs',type=Path,required=True)
    ap.add_argument('--tool',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--benchmark',type=Path,default=Path('notes/m1_m4_recipe.json'))
    ap.add_argument('--phase',choices=('export','capture','validate'),default='capture')
    args=ap.parse_args()
    if args.phase=='export': return export_programs(args)
    if args.phase=='validate': return validate_mil(args)
    if args.out.exists(): raise ValueError('Use a fresh output directory')
    args.out.mkdir(parents=True)
    import macos_m4_environment
    from macos_m4_coreml_runtime import FullANEStack,generate
    from transformers import AutoTokenizer
    from bench_macos_dflash import device_locks,PROMPTS
    result={'schema':1,'status':'waiting','benchmark_sha256':sha256(args.benchmark),
            'golden_runtime':'CoreML CPU_AND_NE, verified all math ANE',
            'hwx_runtime':'_ANEClient precompiled HWX, no CoreML recompilation',
            'linux_replay_verified':False,'cases':[],'generations':[],'programs':{}}
    def save(): (args.out/'capture.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    with device_locks():
        result['status']='capturing';save()
        stack=FullANEStack(args.artifacts)
        tokenizer=AutoTokenizer.from_pretrained(stack.manifest['target'])
        programs=[*(p for _,p in stack.chunks),stack.projector,stack.draft,stack.head]
        for program in programs:
            name=program.role.replace('target','target_')
            root=args.programs/name
            status=plistlib.loads((root/'hwx/model.hwx.status.plist').read_bytes())
            network=status['NetworkStatusList'][0]
            if status['ErrorList']: raise RuntimeError(status['ErrorList'])
            result['programs'][name]={'hwx_sha256':sha256(root/'hwx/model.hwx'),
                                    'original_mil_sha256':sha256(root/'original.mil'),
                                    'network':network}
            original=program.predict
            def predict(inputs,role=None,*,program=program,name=name,network=network,original=original):
                offset=stack.draft_offset if name=='projector' else stack.offset
                label=role or program.role
                stage='early' if (label,'early') not in captured else (
                    'decode' if offset>=prompt_length and (label,'decode') not in captured else (
                    'late' if offset>=80 and (label,'late') not in captured else None))
                # Capture inputs before prediction: cache arrays are mutable views.
                packed={d['Name']:pack_tensor(inputs[d['Name']],d) for d in network['LiveInputList']} if stage else {}
                output=original(inputs,role)
                if stage:
                    captured.add((label,stage))
                    case=args.out/f'{prompt_name}-{label}-{stage}'
                    case.mkdir()
                    row={'program':name,'prompt':prompt_name,'label':label,'stage':stage,'offset':offset,
                         'inputs':[],'outputs':[],'coreml_golden':True,'exact_exported_hwx_evaluated':False}
                    for direction,key,values in [('inputs','LiveInputList',inputs),('outputs','LiveOutputList',output)]:
                        for index,d in enumerate(network[key]):
                            tensor_name=d['Name'].removesuffix('@output')
                            raw=packed[tensor_name] if direction=='inputs' else pack_tensor(values[tensor_name],d)
                            file=f'{direction}-{index}.bin'
                            (case/file).write_bytes(raw)
                            row[direction].append({'name':tensor_name,'index':index,'file':file,
                                                   'shape':list(values[tensor_name].shape),'dtype':'<f2',
                                                   'allocation_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                                                   'compiler_geometry':d})
                    (case/'case.json').write_text(json.dumps(row,indent=2)+'\n')
                    result['cases'].append({'directory':case.name,**row});save()
                return output
            program.predict=predict
        benchmark=json.loads(args.benchmark.read_text())
        for prompt_name,prompt in PROMPTS:
            captured=set();prompt_length=len(tokenizer.encode(prompt))
            generation=generate(stack,tokenizer,prompt,100,True)
            expected=[r for r in benchmark['rows'] if r['name']==prompt_name and r['impl']=='ane_lut6_sd']
            if not expected: raise ValueError('No recorded native SD tokens for prompt')
            identity=all(generation['tokens']==r['tokens'] for r in expected)
            result['generations'].append({'name':prompt_name,'prompt':prompt,'prompt_tokens':tokenizer.encode(prompt),
                                          'matches_native_benchmark':identity,**generation})
            save();print('captured',prompt_name,'token_identity',identity,flush=True)
            if not identity: raise RuntimeError('Captured stack differs from native measured target')
        result['status']='complete_coreml_goldens';save()
        print('Run --phase validate in a fresh process to validate standalone MIL on ANE.',flush=True)


if __name__=='__main__':main()
