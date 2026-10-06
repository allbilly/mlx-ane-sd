"""Check real HWX planning failures and padded tensor transport without ANE."""
import ctypes
from contextlib import ExitStack,nullcontext,redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from capture_m1_full_stack import pack_tensor,unpack_tensor
from m1_ane_replay import plan,BOInit,BOFree,Submit
from m1_hwx_decode import container
from run_m1_full_ane_replay import verify_tensors
from package_m1_full_stack import verify_or_extract
from m1_lut6_weights import Checkpoint,reconstruct,sha256
import run_m1_full_ane_replay as runner


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle=Path(__file__).resolve().parent.parent/'artifacts/m1-full-ane'
        cls.manifest=json.loads((cls.bundle/'manifest.json').read_text())
        cls.hwx=gzip.decompress(cls.read_file('programs/projector/kernel.hwx.gz'))
        cls.status=cls.read_file('programs/projector/status.json')
        cls.expected=json.loads(cls.read_file('programs/projector/linux-plan.json'))
        cls.expected['hwx_sha256']=hashlib.sha256(cls.hwx).hexdigest()
    @classmethod
    def read_file(cls,name):
        recipe=cls.manifest['files'][name];raw=(cls.bundle/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=recipe['sha256']:raise ValueError('Bad reference file')
        return bytes(raw)
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'hwx').mkdir()
        (self.root/'hwx/model.hwx').write_bytes(self.hwx);(self.root/'status.json').write_bytes(self.status)
    def test_real_full_projector_plan_matches_committed_binding_contract(self):
        self.assertEqual(plan(self.root),self.expected)
        self.assertEqual({i['name'] for i in self.expected['io']},{'cos','sin','features','new_k','new_v'})
    def test_aliased_request_surfaces_are_rejected(self):
        s=json.loads(self.status);s['NetworkStatusList'][0]['LiveInputList'].append(s['NetworkStatusList'][0]['LiveInputList'][0])
        (self.root/'status.json').write_text(json.dumps(s))
        with self.assertRaisesRegex(ValueError,'surfaces alias'):plan(self.root)
    def test_interior_bar_cannot_be_silently_bound_to_segment_base(self):
        c=container(self.hwx)
        command=next(cmd for cmd in c['load_commands'] if cmd['command']==4 and struct.unpack_from('<I',bytes.fromhex(cmd['raw_hex']),8)[0]==1)
        raw=bytearray(self.hwx);pos=command['offset']+16+4*8
        struct.pack_into('<Q',raw,pos,struct.unpack_from('<Q',raw,pos)[0]+64)
        (self.root/'hwx/model.hwx').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'interior or alias'):plan(self.root)
    def test_driver_constant_delimiter_must_follow_all_tasks(self):
        c=container(self.hwx)
        command=next(cmd for cmd in c['load_commands'] if cmd['command']==4 and struct.unpack_from('<I',bytes.fromhex(cmd['raw_hex']),8)[0]==1)
        raw=bytearray(self.hwx);base=struct.unpack_from('<Q',raw,command['offset']+16)[0]
        struct.pack_into('<Q',raw,command['offset']+24,base+16)
        (self.root/'hwx/model.hwx').write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'BAR1 delimiter'):plan(self.root)
    def test_padded_causal_mask_transport_preserves_values_and_zeroes_padding(self):
        d={'Type':'Float16','Interleave':1,'Batches':1,'Channels':1,'Depth':1,'Height':8,'Width':264,
           'BatchStride':4608,'PlaneStride':4608,'DepthStride':4608,'RowStride':576}
        mask=np.full((1,1,8,264),-10000,np.float16);mask[0,0,0,0]=0
        raw=pack_tensor(mask,d)
        self.assertEqual(unpack_tensor(raw,d,mask.shape).tobytes(),mask.tobytes())
        self.assertTrue(all(raw[i*576+528:(i+1)*576]==bytes(48) for i in range(8)))
        mask[0,0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'nonfinite'):pack_tensor(mask,d)
    def test_driver_structure_sizes_match_ane_accel_abi(self):
        self.assertEqual((ctypes.sizeof(BOInit),ctypes.sizeof(BOFree),ctypes.sizeof(Submit)),(24,8,152))
    def test_regenerated_input_mismatch_stops_golden_verification(self):
        d={'Type':'Float16','Interleave':1,'Batches':1,'Channels':1,'Depth':1,'Height':1,'Width':8,
           'BatchStride':64,'PlaneStride':64,'DepthStride':64,'RowStride':64}
        value=np.zeros((1,1,1,8),np.float16)
        expected=[{'name':'mask','shape':list(value.shape),'compiler_geometry':d,
                   'sha256':hashlib.sha256(pack_tensor(value,d)).hexdigest()}]
        verify_tensors({'mask':value},expected)
        value[0,0,0,0]=1
        with self.assertRaisesRegex(RuntimeError,'Regenerated tensor differs'):
            verify_tensors({'mask':value},expected)
    def test_compact_bundle_contains_no_learned_banks_or_tensor_binaries(self):
        manifest=verify_or_extract(self.bundle)
        self.assertLess(manifest['stored_bytes'],16*1024*1024)
        self.assertFalse(manifest['learned_weight_banks_included'])
        self.assertFalse(manifest['captured_tensor_binaries_included'])
        recipe=json.loads(self.read_file('programs/projector/packing.json'))
        for op in recipe['operations']:
            self.assertFalse(any(self.hwx[op['offset']:op['offset']+op['size']]))
    def test_external_projector_reconstructs_complete_captured_hwx(self):
        draft=self.bundle.parent.parent/'.asahi/models/microcycle-dflash'
        if not (draft/'model.safetensors').exists():self.skipTest('Pinned draft not downloaded locally')
        stack=json.loads(self.read_file('stack.json'))
        reader=Checkpoint(draft,stack['draft']['safetensors_sha256']);self.addCleanup(reader.close)
        root=self.bundle/'programs/projector'
        recipe=json.loads((root/'packing.json').read_text())
        raw=reconstruct(root,recipe,{'draft':reader})
        self.assertEqual(hashlib.sha256(raw).hexdigest(),recipe['sha256'])
        self.assertNotEqual(raw,self.hwx)
    def test_changed_external_checkpoint_is_rejected(self):
        (self.root/'model.safetensors').write_bytes(b'not the pinned model')
        with self.assertRaisesRegex(ValueError,'Checkpoint differs'):
            Checkpoint(self.root,'0'*64)


class BenchmarkTests(unittest.TestCase):
    def exercise_benchmark(self,warmup_error=None):
        """Exercise the CLI/receipt flow with a cold GPU and no ANE device."""
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);receipt=root/'bench.json'
        bundle=Path(__file__).resolve().parent.parent/'artifacts/m1-full-ane'
        stack=json.loads((bundle/'stack.json').read_text())
        plans={info['program']:{'hwx_sha256':info['hwx_sha256']} for info in stack['artifacts']}
        events=[];cold=True
        def record(label,prompt,limit):
            status=json.loads(receipt.read_text())['status']
            events.append((label,prompt,limit,status))
            return status
        def stock(model,tokenizer,prompt,limit):
            nonlocal cold
            record('mlx_bf16',prompt,limit)
            if warmup_error:raise warmup_error
            rate=1.0 if cold else 100.0;cold=False
            return {'tokens':[7]*limit,'generation_tokens':limit,'generation_tps':rate}
        def ane(host,tokenizer,prompt,limit,speculative):
            record('ane_lut6_sd' if speculative else 'ane_lut6_ar',prompt,limit)
            return {'tokens':[7]*limit,'tok_per_s_decode':100.0}
        def verified(root,programs,capture,values,result,save):
            result['goldens_verified']=True;save()
        model=Mock();tokenizer=SimpleNamespace(decode=lambda ids:str(ids))
        mlx_lm=ModuleType('mlx_lm');mlx_lm.load=Mock(return_value=(model,tokenizer))
        argv=['run_m1_full_ane_replay.py',str(bundle),'--mode','bench',
              '--target',str(root/'target'),'--draft',str(root/'draft'),
              '--cache',str(root/'cache'),'--max-new','12','--repeats','2','--out',str(receipt)]
        with ExitStack() as context:
            context.enter_context(patch('sys.argv',argv))
            context.enter_context(patch.dict('sys.modules',{'mlx_lm':mlx_lm}))
            for name,replacement in {
                'check_sources':Mock(return_value=(stack,plans)),
                'device_locks':lambda:nullcontext(),'device_path':lambda requested:root/'device',
                'prepare':Mock(return_value=root/'cache'),'embedding':Mock(return_value=None),
                'Program':Mock(),'verify_cases':verified,'FullANEStack':Mock(),
                'stock_baseline':stock,'generate':ane,
            }.items():context.enter_context(patch.object(runner,name,replacement))
            context.enter_context(patch.object(runner,'os',SimpleNamespace(
                open=Mock(return_value=99),close=Mock(),O_RDWR=runner.os.O_RDWR,O_CLOEXEC=runner.os.O_CLOEXEC)))
            context.enter_context(redirect_stdout(io.StringIO()))
            context.enter_context(patch('bench_macos_dflash.generate',side_effect=AssertionError('Custom MLX baseline used')))
            if warmup_error:
                with self.assertRaisesRegex(RuntimeError,str(warmup_error)):runner.main()
            else:runner.main()
        return json.loads(receipt.read_text()),events,model
    def test_cold_stock_baseline_is_warmed_and_excluded_from_speedup(self):
        result,events,model=self.exercise_benchmark()
        self.assertEqual(result['status'],'complete');model.eval.assert_called_once()
        self.assertEqual(len(result['warmup']['calls']),12)
        self.assertTrue(result['warmup']['excluded_from_rows'])
        self.assertEqual([event[3] for event in events],['warming']*12+['benchmarking']*24)
        self.assertEqual([event[2] for event in events],[10]*12+[12]*24)
        self.assertEqual({(row['name'],row['impl']) for row in result['warmup']['calls']},
                         {(row['name'],row['impl']) for row in result['rows']})
        self.assertEqual(len(result['rows']),24)
        baseline=[row for row in result['rows'] if row['impl']=='mlx_bf16']
        self.assertTrue(all(row['generation_tps']==row['tok_per_s_decode']==100.0 for row in baseline))
        self.assertEqual(result['sd_vs_linux_mlx'],1.0)
        self.assertIn('stream_generate',result['mlx_baseline'])
        self.assertIn('first token excluded',result['timing_by_impl']['mlx_bf16'])
    def test_failed_warmup_writes_failure_without_measured_rows(self):
        result,events,_=self.exercise_benchmark(RuntimeError('GPU warm-up failed'))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['error'],'GPU warm-up failed')
        self.assertEqual(result['rows'],[])
        self.assertEqual(result['warmup']['calls'],[])
        self.assertEqual([event[3] for event in events],['warming'])


if __name__=='__main__':unittest.main()
