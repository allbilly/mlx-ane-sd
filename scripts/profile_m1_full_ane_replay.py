"""Separate host transport and synchronous submit costs for real M1 SD traces.

This diagnostic wraps the existing replay implementation without changing its
programs, buffers or arithmetic. Its timings are separate from benchmark rates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from bench_macos_dflash import device_locks
from capture_m1_full_stack import sha256
import m1_ane_replay as replay
from m1_full_ane_host import FullANEStack, generate,trace_matches
from run_m1_full_ane_replay import check_sources, embedding


class Profiler:
    def __init__(self):
        self.active = None
        self.samples = []

    def timed(self, key, operation, *args):
        begin = time.perf_counter_ns()
        try:
            return operation(*args)
        finally:
            if self.active is not None:
                self.active[key] = self.active.get(key, 0) + time.perf_counter_ns() - begin

    def exclusive(self, key, operation, *args):
        row=self.active
        before=sum(value for name,value in (row or {}).items() if name.endswith('_ns'))
        begin=time.perf_counter_ns()
        try:return operation(*args)
        finally:
            if row is not None:
                nested=sum(value for name,value in row.items() if name.endswith('_ns'))-before
                row[key]=row.get(key,0)+time.perf_counter_ns()-begin-nested


class ReadTimedMap:
    def __init__(self, mapping, profiler):
        self.mapping, self.profiler = mapping, profiler

    def __getitem__(self, key):
        return self.profiler.timed('mapped_read_ns', self.mapping.__getitem__, key)

    def __setitem__(self, key, value):
        self.mapping[key] = value

    def close(self):
        self.mapping.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    ap.add_argument('--runtime', type=Path, default=Path('.asahi/m1-full-ane-cache'))
    ap.add_argument('--target', type=Path, required=True)
    ap.add_argument('--transport',choices=('reference','packed','resident'),default='resident')
    ap.add_argument('--head-readback',choices=('full','rows','mapped','native'),default='mapped')
    ap.add_argument('--verify-readback',choices=('all','accepted-prefix'),default='all')
    ap.add_argument('--kv-readback',choices=('full','prefix'),default='full')
    ap.add_argument('--native-threads',type=int,default=1)
    ap.add_argument('--cpu-affinity')
    ap.add_argument('--cpu-util-min',type=int)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cpu_hint=None
    if args.cpu_util_min is not None:
        from m1_scheduler import utilization_minimum
        cpu_hint=utilization_minimum(args.cpu_util_min)
    if not 1<=args.native_threads<=32:ap.error('--native-threads must be in 1..32')
    if args.cpu_affinity:
        affinity={int(cpu) for cpu in args.cpu_affinity.split(',')}
        if not affinity or not affinity<=os.sched_getaffinity(0):ap.error('--cpu-affinity must select available CPUs')
        os.sched_setaffinity(0,affinity)
    if args.out.exists():
        ap.error('--out must be fresh')
    stack, _ = check_sources(args.runtime, templates=False)
    capture = json.loads((args.root / 'cases/capture.json').read_text())
    profiler, programs, roles = Profiler(), {}, {}
    originals = {key: getattr(replay, key) for key in ('pack_tensor', 'unpack_tensor', 'ioctl')}
    original_write = replay.Buffer.write
    original_write_at = replay.Buffer.write_at
    original_stage = replay.InputTransport.write
    original_rows = replay.InputTransport.write_rows
    original_reduce = replay.Program.read_token_ids
    original_prefix = replay.CacheOutput.read_prefix

    def pack(*values):
        return profiler.timed('pack_ns', originals['pack_tensor'], *values)

    def unpack(*values):
        return profiler.timed('unpack_ns', originals['unpack_tensor'], *values)

    def ioctl(fd, command, request):
        begin=time.thread_time_ns()
        try:return profiler.timed('submit_ns', originals['ioctl'], fd, command, request)
        finally:
            if profiler.active is not None:
                # CPU time overlaps the wall span, so keep it outside the
                # exclusive *_ns fields used by the accounting below.
                row=profiler.active
                row['submit_cpu_ms']=row.get('submit_cpu_ms',0)+(time.thread_time_ns()-begin)/1e6

    def write(buffer, data):
        role = roles.get(id(buffer), 'other')
        if profiler.active is not None:
            key = role + '_write_bytes'
            profiler.active[key] = profiler.active.get(key, 0) + len(data)
        return profiler.timed(role + '_write_ns', original_write, buffer, data)

    def write_at(buffer,data,offset):
        role=roles.get(id(buffer),'other')
        if profiler.active is not None:
            key=role+'_write_bytes'
            profiler.active[key]=profiler.active.get(key,0)+len(data)
        return profiler.timed(role+'_write_ns',original_write_at,buffer,data,offset)

    def stage(transport,values,buffer):
        return profiler.exclusive('pack_ns',original_stage,transport,values,buffer)

    def rows(transport,values,offset,count,buffer):
        return profiler.exclusive('pack_ns',original_rows,transport,values,offset,count,buffer)

    def reduce(program,start,end,outputs=None):
        # In mapped mode this includes reading/converting logits and argmax.
        return profiler.exclusive('head_reduce_ns',original_reduce,program,start,end,outputs)

    def prefix(output,count):
        row={'prompt':current_prompt,'program':output.program.root.name,'role':'kv_read','event':'kv_read'}
        shape,_,_=replay.geometry(output.tensor['compiler_geometry'])
        row['requested_tensor_bytes']=shape[0]*shape[1]*shape[2]*count*shape[4]*2
        profiler.active=row;begin=time.perf_counter_ns()
        try:return original_prefix(output,count)
        finally:
            row['total_ns']=time.perf_counter_ns()-begin
            profiler.samples.append(row);profiler.active=None

    class Measured:
        def __init__(self, info):
            self.info, self.program = info, programs[info['program']]

        def reset_transport(self):self.program.reset_transport()

        def commit_cache(self,*values):
            return self.sample('cache_update',self.program.commit_cache,values)

        def predict(self, inputs, role=None):
            return self.sample('predict',self.program.predict,(inputs,),role)

        def __getattr__(self,name):
            if name=='predict_token_ids' and args.head_readback!='full':return self.tokens
            if name=='predict_verified_ids' and args.verify_readback=='accepted-prefix':return self.verified
            raise AttributeError(name)

        def tokens(self,inputs,start,end,role=None):
            return self.sample('predict',self.program.predict_token_ids,(inputs,start,end),role)

        def verified(self,inputs,candidates,start,end,role=None):
            return self.sample('predict',self.program.predict_verified_ids,(inputs,candidates,start,end),role)

        def sample(self,event,operation,values,role=None):
            row = {'prompt': current_prompt, 'program': self.info['program'],
                   'role': role or self.info['role'],'event':event}
            profiler.active = row
            begin = time.perf_counter_ns()
            try:
                return operation(*values)
            finally:
                row['total_ns'] = time.perf_counter_ns() - begin
                accounted = sum(value for key, value in row.items()
                                if key.endswith('_ns') and key != 'total_ns')
                row['other_ns'] = row['total_ns'] - accounted
                profiler.samples.append(row)
                profiler.active = None

    result = {'schema': 1, 'status': 'starting', 'procedure': __doc__, 'generations': [],
              'transport':args.transport,'head_readback':args.head_readback,
              'verify_readback':args.verify_readback,'kv_readback':args.kv_readback,
              'cpu_affinity':sorted(os.sched_getaffinity(0)),
              'cpu_utilization_hint':cpu_hint,
              'source_sha256': {name: sha256(Path(__file__).parent / name) for name in
                                (Path(__file__).name, 'm1_ane_replay.py', 'm1_full_ane_host.py')},
              'program_sha256': {info['program']: info['hwx_sha256'] for info in stack['artifacts']},
              'submit_timing': 'synchronous ioctl span; includes device execution and driver wait'}
    if args.head_readback=='native':
        from m1_native_transport import library,configure
        configure(args.native_threads)
        result['native_transport']=library()[1]
        result['native_threads']=args.native_threads
        result['source_sha256']['m1_native_transport.py']=sha256(Path(__file__).parent/'m1_native_transport.py')
    if cpu_hint is not None:
        result['source_sha256']['m1_scheduler.py']=sha256(Path(__file__).parent/'m1_scheduler.py')

    def save():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')

    save()
    fd = None
    try:
        with device_locks():
            values = embedding(args.target, stack)
            fd = os.open(replay.device_path(), os.O_RDWR | os.O_CLOEXEC)
            for info in stack['artifacts']:
                name = info['program']
                example = next(row for row in capture['cases'] if row['program'] == name)
                shapes = {tensor['name']: tensor['shape'] for tensor in example['outputs']}
                program = programs[name] = replay.Program(fd, args.runtime / 'programs' / name, shapes,
                                                          transport=args.transport,head_readback=args.head_readback,
                                                          verify_readback=args.verify_readback,kv_readback=args.kv_readback)
                io_roles = {tensor['bank']: tensor['direction'].removesuffix('s') for tensor in program.meta['io']}
                for bank, buffer in program.buffers.items():
                    role = io_roles.get(bank, next((b['role'] for b in program.meta['buffers'] if b['bank'] == bank), 'command'))
                    roles[id(buffer)] = role
                    buffer.map = ReadTimedMap(buffer.map, profiler)
            replay.pack_tensor, replay.unpack_tensor, replay.ioctl = pack, unpack, ioctl
            replay.Buffer.write = write
            replay.Buffer.write_at=write_at
            replay.InputTransport.write,replay.InputTransport.write_rows=stage,rows
            replay.Program.read_token_ids=reduce
            replay.CacheOutput.read_prefix=prefix
            host = FullANEStack(args.root, values, Measured)

            class Tokenizer:
                eos_token_ids = {host.cfg['eos_token_id']}

                def encode(self, prompt):
                    return next(g['prompt_tokens'] for g in capture['generations'] if g['prompt'] == prompt)

                def decode(self, tokens):
                    return ''

            result['status'] = 'profiling'
            save()
            for golden in capture['generations']:
                current_prompt = golden['name']
                actual = generate(host, Tokenizer(), golden['prompt'], 100, True)
                matched = actual['tokens'] == golden['tokens'] and trace_matches(actual['trace'],golden['trace'])
                result['generations'].append({'prompt': current_prompt, 'tokens': len(actual['tokens']),
                                              'matches_macOS_trace': matched,
                                              'tokens_sha256': hashlib.sha256(bytes(str(actual['tokens']), 'ascii')).hexdigest()})
                if not matched:
                    raise RuntimeError('Profiling changed the captured SD trace')
                save()
            for event,key in (('predict','programs'),('cache_update','cache_updates'),('kv_read','kv_reads')):
                result[key]={}
                for name in programs:
                    rows = [row for row in profiler.samples if row['program'] == name and row['event']==event]
                    if not rows:continue
                    fields = sorted({field for row in rows for field in row if field.endswith(('_ns', '_bytes'))})
                    result[key][name] = {
                        'calls': len(rows),
                        'mean_ms': {field.removesuffix('_ns'): statistics.mean(row.get(field, 0) for row in rows) / 1e6
                                    for field in fields if field.endswith('_ns')},
                        'max_ms': {field.removesuffix('_ns'): max(row.get(field, 0) for row in rows) / 1e6
                                   for field in fields if field.endswith('_ns')},
                        'mean_bytes_per_call': {field:statistics.mean(row.get(field,0) for row in rows)
                                                for field in fields if field.endswith('_bytes')},
                        'per_prompt_mean_ms': {g['name']: statistics.mean(row['total_ns'] for row in rows if row['prompt'] == g['name']) / 1e6
                                               for g in capture['generations']},
                    }
                    if any('submit_cpu_ms' in row for row in rows):
                        result[key][name]['mean_submit_thread_cpu_ms']=statistics.mean(row.get('submit_cpu_ms',0) for row in rows)
            result['status'] = 'complete'
            save()
            print(json.dumps(result['programs'], indent=2))
    except BaseException as error:
        result['status'], result['error'] = 'failed', str(error)
        save()
        raise
    finally:
        for key, value in originals.items():
            setattr(replay, key, value)
        replay.Buffer.write = original_write
        replay.Buffer.write_at=original_write_at
        replay.InputTransport.write,replay.InputTransport.write_rows=original_stage,original_rows
        replay.Program.read_token_ids=original_reduce
        replay.CacheOutput.read_prefix=original_prefix
        for program in programs.values():
            program.close()
        if fd is not None:
            os.close(fd)


if __name__ == '__main__':
    main()
