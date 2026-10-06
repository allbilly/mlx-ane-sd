"""Check regenerated golden inputs/outputs against the measured macOS stack.

Uses existing compiled CoreML artifacts with verified ANE math placement.
This validates the compact kit's host/input regeneration; it neither executes
the offline HWX nor measures a new speedup. Linux has its own verifier.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import macos_m4_environment  # Read-only environment bridge before NumPy/CoreML.
from bench_macos_dflash import device_locks
from m1_lut6_weights import sha256
from run_m1_full_ane_replay import check_sources,embedding,verify_cases


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--artifacts',type=Path,required=True)
    ap.add_argument('--target',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():ap.error('Use a fresh receipt path')
    stack,_=check_sources(args.root)
    source=json.loads((args.artifacts/'manifest.json').read_text())
    for info in source['artifacts']:
        name=f"target_{info['start']}" if info['role']=='target' else info['role']
        expected=next(row for row in stack['artifacts'] if row['program']==name)
        compiled=Path(info['compiled'])
        if sha256(compiled/'model.mil')!=expected['mil_sha256'] or sha256(compiled/'weights/weight.bin')!=expected['weights_sha256']:
            raise ValueError('Compiled macOS source differs from captured reference')
    result={'status':'waiting','purpose':'Regenerate inputs and verify captured hashes on macOS CoreML ANE; not a timing benchmark',
            'linux_replay_verified':False,'offline_hwx_evaluated':False,'cases':[],
            'source_sha256':{name:sha256(Path(__file__).parent/name) for name in
                ('verify_m1_full_ane_macos.py','run_m1_full_ane_replay.py','m1_full_ane_host.py','m1_lut6_weights.py')},
            'bundle_manifest_sha256':sha256(args.root/'manifest.json')}
    def save():
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        with device_locks():
            from macos_m4_coreml_runtime import FullANEStack
            reference=FullANEStack(args.artifacts)
            programs={f"target_{info['start']}":program for info,program in reference.chunks}
            programs.update(projector=reference.projector,draft=reference.draft,head=reference.head)
            capture=json.loads((args.root/'cases/capture.json').read_text())
            result['status']='verifying';save()
            verify_cases(args.root,programs,capture,embedding(args.target,stack),result,save)
            result['status']='complete';save()
            print('Verified',len(result['cases']),'calls and',sum(len(row['checks']) for row in result['cases']),'outputs')
    except BaseException as error:
        result.update(status='failed',error=str(error));save();raise


if __name__=='__main__':main()
