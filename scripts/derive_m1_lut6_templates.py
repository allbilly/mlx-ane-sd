"""Derive and byte-verify weight-free templates from local macOS captures.

Reads the large captures locally. Only templates, scalar quantizer recipes,
MIL source and capture hashes are emitted. Never packages checkpoint banks,
packed coefficient banks, embedding or captured tensor data.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import numpy as np
from m1_hwx_decode import container
from m1_lut6_weights import Checkpoint, group_payloads, matrix_bytes, palette_means, quantize, reconstruct, require, sha256, unpack6

LUT = re.compile(r'tensor<fp16, \[([^]]+)\]> (\w+) = constexpr_lut_to_dense\(indices = tensor<uint6, \[[^]]+\]>\(BLOBFILE\(path = string\("[^"]+"\), offset = uint64\((\d+)\)\)\), lut = tensor<fp16, \[([^]]+)\]>\(BLOBFILE\(path = string\("[^"]+"\), offset = uint64\((\d+)\)\)\)\)')
VECTOR = re.compile(r'tensor<fp16, \[([^]]+)\]> (\w+) = const\(\).*?offset = uint64\((\d+)\)')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def blob(data, offset, dtype):
    magic, actual, size, begin = struct.unpack_from('<IIQQ', data, int(offset))
    require(magic == 0xDEADBEEF and actual == dtype and begin + size <= len(data), 'Invalid MIL BLOB descriptor')
    return data[begin:begin + size]


def tensor_ref(name, info):
    role = info['role']
    if role == 'head':
        tile = int(re.match(r'tiles_(\d+)_weight', name)[1])
        return 'target', 'model.embed_tokens.weight', [tile * 8192, min((tile + 1) * 8192, 151936)]
    if role == 'projector':
        if name.startswith('fc_'):
            return 'draft', 'fc.weight', None
        match = re.match(r'([kv])_(\d+)_weight', name)
        return 'draft', f'layers.{match[2]}.self_attn.{match[1]}_proj.weight', None
    if name == 'norm_weight':
        return 'target', 'model.norm.weight', None
    match = re.match(r'layers_(\d+)_(input_ln|post_ln|q_norm|k_norm|q|k|v|o|gate|up|down)_weight', name)
    require(match is not None, 'Unknown compiler tensor: ' + name)
    index, kind = int(match[1]) + info.get('start', 0), match[2]
    suffix = {'input_ln': 'input_layernorm', 'post_ln': 'post_attention_layernorm',
              'q_norm': 'self_attn.q_norm', 'k_norm': 'self_attn.k_norm'}
    suffix.update({k: f'self_attn.{k}_proj' for k in ('q', 'k', 'v', 'o')})
    suffix.update({k: f'mlp.{k}_proj' for k in ('gate', 'up', 'down')})
    prefix = 'model.' if role == 'target' else ''
    return ('target' if role == 'target' else 'draft'), f'{prefix}layers.{index}.{suffix[kind]}.weight', None


def derive_program(source, destination, info, checkpoints):
    destination.mkdir(parents=True,exist_ok=True)
    data = (source / 'hwx/model.hwx').read_bytes()
    require(hashlib.sha256(data).hexdigest()==info['hwx_sha256'],'Source HWX differs from captured stack')
    require(sha256(source/'original.mil')==info['mil_sha256'] and sha256(source/'weights/weight.bin')==info['weights_sha256'],
            'Source MIL/weights differ from measured compiler inputs')
    if (destination/'packing.json').exists():
        recipe=json.loads((destination/'packing.json').read_text())
        require(reconstruct(destination,recipe,checkpoints)==data,'Changed resumed program')
        return program_result(info,recipe,destination)
    original = (source / 'weights/weight.bin').read_bytes()
    mil = (source / 'original.mil').read_text()
    template = bytearray(data)
    c = container(data)
    palette_locations = {}
    for symbol in c['symbols']:
        if '_ne_' not in symbol['name']:
            continue
        section = c['sections'][symbol['section'] - 1]
        base = section['offset'] + symbol['value'] - section['addr']
        for begin in (base, base + 128):
            palette_locations.setdefault(data[begin:begin + 128], []).append(begin)
    boundaries = bytearray()
    operations = []
    for shape, name, index_offset, palette_shape, palette_offset in LUT.findall(mil):
        shape = [int(v) for v in shape.split(',')]
        indices = unpack6(blob(original, index_offset, 13), shape)
        palettes = np.frombuffer(blob(original, palette_offset, 1), '<f2').reshape(-1, 64)
        group_rows = 0 if len(palettes) == 1 else shape[0] // len(palettes)
        require(group_rows in (0, 16), 'Unexpected quantizer granularity')
        checkpoint, tensor, rows = tensor_ref(name, info)
        values = checkpoints[checkpoint].read(tensor, rows)
        require(list(values.shape) == shape, 'Source checkpoint shape mismatch')
        operation = dict(kind='lut6', checkpoint=checkpoint, tensor=tensor, shape=shape,
                         group_rows=group_rows, boundary_offset=len(boundaries), palette_corrections=[])
        if rows:
            operation['rows'] = rows
        for group in range(len(palettes)):
            selection = slice(None) if group_rows == 0 else slice(group * group_rows, (group + 1) * group_rows)
            w, ids = values[selection].reshape(-1), indices[selection].reshape(-1)
            minima, maxima = np.full(64, np.inf, np.float32), np.full(64, -np.inf, np.float32)
            np.minimum.at(minima, ids, w)
            np.maximum.at(maxima, ids, w)
            require(bool(np.isfinite(minima).all() and np.isfinite(maxima).all() and
                         (maxima[:-1] < minima[1:]).all()), 'Clusters cannot be reproduced by scalar boundaries: ' + name)
            cuts = maxima[:-1].astype('<f2')
            require(np.array_equal(np.searchsorted(cuts, w, side='left'), ids), 'Boundary reconstruction differs')
            boundaries.extend(cuts.tobytes())
            means = palette_means(w, ids)
            for index in np.flatnonzero(means.view('<u2') != palettes[group].view('<u2')):
                operation['palette_corrections'].append([group, int(index), int(palettes[group].view('<u2')[index])])
        if not operation['palette_corrections']:
            del operation['palette_corrections']
        # Match the complete coefficient payload, including engine/group order.
        for tile in ((32,) if group_rows else (32, 16)):
            payload = matrix_bytes(indices, palettes, group_rows, tile)
            offset = data.find(payload)
            if offset >= 0:
                if not group_rows:
                    operation['tile'] = tile
                break
        else:
            for tile in ((32,) if group_rows else (32,16)):
                if group_rows:
                    parts=group_payloads(indices,palettes)
                else:
                    payload=matrix_bytes(indices,palettes,0,tile)
                    width=len(payload)//16
                    parts=[payload[index*width:(index+1)*width] for index in range(16)]
                locations=[]
                for index,part in enumerate(parts):
                    candidates=sorted(set(palette_locations.get(part[:128],[]))-set(locations))
                    matches=[begin for begin in candidates if data[begin:begin+len(part)]==part]
                    if not matches:break
                    locations.append(matches[0])
                if len(locations)==len(parts):break
            require(len(locations)==len(parts),'Missing/ambiguous H13 engine layout: '+name)
            operation['group_offsets' if group_rows else 'engine_offsets']=locations
            if not group_rows:operation['tile']=tile
            payload = b''.join(parts)
        if 'group_offsets' in operation or 'engine_offsets' in operation:
            extents = zip(operation.get('group_offsets',operation.get('engine_offsets')), parts)
        else:
            require(data.find(payload, offset + 1) < 0, 'Ambiguous learned payload')
            operation['offset'] = offset
            extents = [(offset, payload)]
        operation.update(size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        for begin, part in extents:
            require(any(template[begin:begin + len(part)]), 'Duplicate payload')
            template[begin:begin + len(part)] = bytes(len(part))
        check_ids, check_palettes = quantize(values, operation, boundaries)
        require(np.array_equal(check_ids, indices) and check_palettes.tobytes() == palettes.tobytes(), 'Quantizer differs')
        operations.append(operation)
        print(info['program'], name, 'verified', flush=True)
    for shape, name, offset in VECTOR.findall(mil):
        payload = blob(original, offset, 1)
        if not any(payload):
            continue  # Compiler-generated zero biases are fixed constants.
        checkpoint, tensor, rows = tensor_ref(name, info)
        values = checkpoints[checkpoint].read(tensor, rows)
        require(values.tobytes() == payload, 'Learned vector differs from external checkpoint')
        begin = data.find(payload)
        require(begin >= 0 and data.find(payload, begin + 1) < 0, 'Missing/ambiguous vector')
        operations.append(dict(kind='vector', checkpoint=checkpoint, tensor=tensor,
                               shape=list(values.shape), offset=begin, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
        template[begin:begin + len(payload)] = bytes(len(payload))
    # Audit coefficient sections: residual bytes may only be fixed RMS/rsqrt
    # and exponential activation tables, never an unexplained learned bank.
    literals = []
    for section_index, section in enumerate(c['sections'], 1):
        if not section['name'].startswith('__kern_'):
            continue
        leftover = bytearray(template[section['offset']:section['offset'] + section['size']])
        for symbol in c['symbols']:
            if symbol['section'] != section_index or not symbol['name'].startswith('K') or '_ne_' in symbol['name']:
                continue
            start = symbol['value'] - section['addr']
            size = 2048 if symbol['name'] == 'K429D76FEACD2496DECEF4539D424F14BB60524C582F11B139E0004A9776C14FD' else 128
            raw = bytes(leftover[start:start + size])
            if any(raw):
                require(symbol['name'] in {
                    'K471E3C462A3A0BEE5592270D4C57BEB8AC4A58C74AA26B5ADE308A24590D923C',
                    'K7C876A59CE09916F1FF9DA8B77C438AE332776274882643F6CC820A226AF9AE9',
                    'K70E0CDC78402315626D7DFACAB2A1DCE059EE6EE0BB2A49D92F60B4900B60570',
                    'K0A4633DF9F9F0357D19A9979D321A8B9A1C46820C61971F7E3AC1283EAFDADA9',
                    'K429D76FEACD2496DECEF4539D424F14BB60524C582F11B139E0004A9776C14FD'},
                    'Unexplained coefficient literal: ' + symbol['name'])
                literals.append(dict(symbol=symbol['name'], sha256=hashlib.sha256(raw).hexdigest(), bytes=size))
                leftover[start:start + size] = bytes(size)
        # Fused gate projections carry one identical SiLU table per engine
        # group. This fingerprint is shared by target and draft kernels.
        silu_sha='0d48f9b9a9a791fcef312a765cd621e43acb2b56306b9bf491ee0957c5842897'
        count=0
        for begin in range(0,len(leftover),128):
            raw=bytes(leftover[begin:begin+128])
            if any(raw) and hashlib.sha256(raw).hexdigest()==silu_sha:
                leftover[begin:begin+128]=bytes(128);count+=1
        if count:literals.append(dict(kind='fixed fused SiLU table',sha256=silu_sha,bytes=128,copies=count))
        if any(leftover):
            nonzero=np.flatnonzero(np.frombuffer(leftover,np.uint8))
            samples=[{'offset':section['offset']+int(i),'hex':bytes(leftover[max(0,int(i)-8):int(i)+40]).hex()} for i in nonzero[::max(1,len(nonzero)//5)][:5]]
            raise ValueError('Unexplained coefficient residue: '+info['program']+' '+str(len(nonzero))+' '+str(samples))
    template_path = destination / 'kernel.hwx.gz'
    template_path.write_bytes(gzip.compress(template, compresslevel=9, mtime=0))
    boundary_path = destination / 'quantizer.bin.gz'
    boundary_path.write_bytes(gzip.compress(boundaries, compresslevel=9, mtime=0))
    recipe = dict(schema=1, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                  template=template_path.name, template_sha256=sha256(template_path),
                  boundaries=boundary_path.name, boundaries_sha256=sha256(boundary_path),
                  quantizer_bytes=len(boundaries), learned_coefficients_zeroed=True,
                  fixed_activation_tables=literals, operations=operations)
    dump(destination / 'packing.json', recipe)
    for name in ('status.json', 'linux-plan.json', 'original.mil'):
        shutil.copyfile(source / name, destination / name)
    # Whole-container identity also checks every command, table and BAR.
    require(reconstruct(destination, recipe, checkpoints) == data, 'Full HWX reconstruction differs')
    return program_result(info,recipe,destination)


def program_result(info,recipe,destination):
    operations=recipe['operations']
    return dict(program=info['program'], hwx_sha256=recipe['sha256'],
                matrices=sum(op['kind'] == 'lut6' for op in operations), vectors=sum(op['kind'] == 'vector' for op in operations),
                template_bytes=(destination/recipe['template']).stat().st_size, quantizer_bytes=(destination/recipe['boundaries']).stat().st_size,
                coefficient_bytes=sum(op['size'] for op in operations), full_hwx_byte_identical=True)


def derive(capture, target, draft, out,resume=False):
    require(resume or not out.exists(), 'Use a fresh compact bundle directory')
    goldens=json.loads((capture/'cases/capture.json').read_text())
    require(goldens['status']=='complete_coreml_and_mil_goldens' and len(goldens['cases'])==72 and
            sum(len(row['outputs']) for row in goldens['cases'])==588 and
            all(row.get('mil_evaluated_on_ane') and all(check['byte_identical'] and check['finite'] for check in row['mil_output_checks'])
                for row in goldens['cases']), 'Need all 72 real calls and 588 standalone MIL ANE goldens')
    stack = json.loads((capture / 'stack.json').read_text())
    stack['draft']['safetensors_sha256'] = sha256(draft / 'model.safetensors')
    checkpoints = {}
    try:
        for name, directory in [('target', target), ('draft', draft)]:
            checkpoints[name] = Checkpoint(directory, stack[name]['safetensors_sha256'])
        out.mkdir(parents=True,exist_ok=resume)
        # NumPy versions can cross an fp16 rounding boundary in sin(). Keep
        # this small deterministic math table, rather than bulk input dumps.
        require(np.__version__=='2.4.6','Derive positional constants using the measured NumPy 2.4.6 environment')
        from m1_full_ane_host import rope
        cosine,sine=rope(0,stack['capacity']+stack['block'],stack['config']['head_dim'],stack['config']['rope_theta'])
        half=stack['config']['head_dim']//2
        np.savez_compressed(out/'rope.npz',cos=cosine[0,0,:,:half],sin=sine[0,0,:,:half])
        stack['positional_tables']={'file':'rope.npz','sha256':sha256(out/'rope.npz'),
            'kind':'fixed mathematical RoPE constants, no learned weights or captured model inputs',
            'generation_numpy':'2.4.6','positions':stack['capacity']+stack['block']}
        dump(out / 'stack.json', stack)
        results = []
        for info in stack['artifacts']:
            results.append(derive_program(capture / 'programs' / info['program'], out / 'programs' / info['program'], info, checkpoints))
        (out / 'cases').mkdir(exist_ok=resume)
        shutil.copyfile(capture / 'cases/capture.json', out / 'cases/capture.json')
        dump(out / 'reconstruction.json', dict(status='complete', linux_replay_verified=False,
             original_benchmark_sha256=goldens['benchmark_sha256'], programs=results,
             omitted=['compiler weight.bin', 'learned HWX coefficient banks', 'embedding', 'captured input/output binaries']))
        return results
    finally:
        for checkpoint in checkpoints.values():
            checkpoint.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'target', 'draft', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--resume',action='store_true',help='Recheck completed programs in a partial local derivation')
    args = ap.parse_args()
    print(json.dumps(derive(args.capture, args.target, args.draft, args.out,args.resume), indent=2))
