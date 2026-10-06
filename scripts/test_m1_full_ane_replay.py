"""Check real HWX planning failures and padded tensor transport without ANE."""
import ctypes
import gzip
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from capture_m1_full_stack import pack_tensor,unpack_tensor
from m1_ane_replay import plan,BOInit,BOFree,Submit
from m1_hwx_decode import container
from run_m1_full_ane_replay import verify_tensors
from package_m1_full_stack import verify_or_extract
from m1_lut6_weights import Checkpoint,reconstruct,sha256


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


if __name__=='__main__':unittest.main()
