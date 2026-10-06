"""Plan and replay complete H13G programs through the Asahi ANE DRM ABI.

The planner is usable on macOS and checks every BAR, surface and task before
hardware submission. Linux execution remains unverified until a receipt says
otherwise. ABI reference: allbilly/ane kmod/uapi/drm/ane_accel.h and ane_drv.c.
"""
from __future__ import annotations
import ctypes
from fcntl import ioctl
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import stat
import struct
import numpy as np
from capture_m1_full_stack import geometry,pack_tensor,unpack_tensor
from m1_hwx_decode import container,tasks,bounds


def require(condition,message):
    if not condition: raise ValueError(message)


def plan(root,data=None):
    root=Path(root)
    if data is None:data=(root/'hwx/model.hwx').read_bytes()
    c=container(data)
    t=next(t for t in c['threads'] if 'bars' in t)
    text=next(s for s in c['segments'] if s['name']=='__TEXT')
    require(t['entry']==text['vmaddr']==t['bars'][0],'Expected entry at command BAR zero')
    command=bounds(data,text['fileoff'],text['filesize'])
    td,writes=tasks(command,0,t['first_td_size'],t['td_count'])
    split=t['bars'][1]-t['bars'][0]
    require(split%16==0 and td[-1]['offset']+td[-1]['size']<=split<len(command),
            'Cannot reproduce the driver-synthesized BAR1 delimiter')
    require(all(d['network_id']==0 for d in td),'Unsupported multi-network descriptor stream')
    buffers=[]
    for bank,address in enumerate(t['bars']):
        if not address or bank in (0,1): continue
        owners=[(i,s) for i,s in enumerate(c['segments']) if s['vmaddr']==address and s['vmsize']]
        require(len(owners)==1,f'BAR {bank} needs an unsupported interior or alias binding')
        index,s=owners[0]
        require(s['filesize']<=s['vmsize'],'Segment payload larger than allocation')
        buffers.append({'bank':bank,'segment_index':index,'segment':s['name'],
                        'bytes':s['vmsize'],'file_offset':s['fileoff'],'file_bytes':s['filesize'],
                        'role':'scratch' if s['name']=='__DATA' else ('coefficient' if s['filesize'] else 'surface')})
    network=json.loads((root/'status.json').read_text())['NetworkStatusList'][0]
    io=[]
    for direction,key in [('inputs','LiveInputList'),('outputs','LiveOutputList')]:
        for index,d in enumerate(network[key]):
            name=d['Name'].removesuffix('@output')
            symbol=[s for s in c['symbols'] if s['name']==name and s['type']==15]
            require(len(symbol)==1,f'Missing/ambiguous tensor symbol {name}')
            banks=[i for i,a in enumerate(t['bars']) if a==symbol[0]['value'] and a]
            require(len(banks)==1 and banks[0] not in (0,1),f'Aliased tensor BAR: {name}')
            shape,strides,size=geometry(d)
            buffer=next(b for b in buffers if b['bank']==banks[0])
            require(buffer['role']=='surface' and buffer['bytes']>=size,'Bad tensor allocation')
            io.append({'direction':direction,'index':index,'name':name,'bank':banks[0],
                       'allocation_bytes':size,'compiler_geometry':d})
    require(len({i['bank'] for i in io})==len(io),'Input/output surfaces alias')
    used={s['bank'] for task in td for s in task['base_selectors'] if s['enabled']}
    require(used<=({0,1}|{b['bank'] for b in buffers}),'Task references an unallocated BAR')
    require({b['bank'] for b in buffers if b['role']=='surface'}=={i['bank'] for i in io},
            'An external surface has no tensor binding')
    return {'schema':1,'architecture':'H13G','hwx_sha256':hashlib.sha256(data).hexdigest(),
            'td_count':len(td),'td_size':t['first_td_size'],'tsk_size':split,
            'text_file_offset':text['fileoff'],'text_file_bytes':text['filesize'],
            'buffers':buffers,'io':io,'active_bars':sorted(used),
            'bootstrap_network_id':64,'original_network_id':0,
            'driver_synthesizes_bar1':True,'task_descriptors_modified':False,
            'linux_hardware_verified':False}


class BOInit(ctypes.Structure):
    _fields_=[('handle',ctypes.c_uint32),('pad',ctypes.c_uint32),('size',ctypes.c_uint64),('offset',ctypes.c_uint64)]
class BOFree(ctypes.Structure):
    _fields_=[('handle',ctypes.c_uint32),('pad',ctypes.c_uint32)]
class Submit(ctypes.Structure):
    _fields_=[('tsk_size',ctypes.c_uint64),('td_count',ctypes.c_uint32),('td_size',ctypes.c_uint32),
             ('handles',ctypes.c_uint32*32),('btsp_handle',ctypes.c_uint32),('pad',ctypes.c_uint32)]


def iowr(number,cls): return (3<<30)|(0x64<<8)|(ctypes.sizeof(cls)<<16)|number
INIT,FREE,SUBMIT=iowr(0x41,BOInit),iowr(0x42,BOFree),iowr(0x43,Submit)


def device_path(requested=None):
    require(platform.system()=='Linux' and platform.machine() in ('aarch64','arm64'),
            'Hardware replay requires M1 Asahi Linux')
    require(b'apple,t8103\0' in Path('/proc/device-tree/compatible').read_bytes(),'Expected base M1 t8103')
    for p in ([Path(requested)] if requested else sorted(Path('/dev/accel').glob('accel*'))):
        if not p.exists() or not stat.S_ISCHR(p.stat().st_mode): continue
        driver=Path('/sys/class/accel')/p.name/'device/driver'
        if driver.resolve().name=='ane' and os.access(p,os.R_OK|os.W_OK): return p
    raise ValueError('No accessible ANE accel device with the allbilly/libane submit ABI')


class Buffer:
    def __init__(self,fd,size):
        self.fd,self.handle,self.map=fd,0,None
        self.size=(size+16383)&~16383
        request=BOInit(size=self.size);ioctl(fd,INIT,request)
        self.handle=request.handle
        require(self.handle!=0,'Driver returned a zero buffer handle')
        try:self.map=mmap.mmap(fd,self.size,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE,offset=request.offset)
        except BaseException:self.close();raise
    def write(self,data):
        require(len(data)<=self.size,'Surface overflow');self.map[:len(data)]=data
    def write_at(self,data,offset):
        require(offset>=0 and offset+len(data)<=self.size,'Surface overflow')
        self.map[offset:offset+len(data)]=data
    def close(self):
        if self.map is not None:self.map.close();self.map=None
        if self.handle:
            try:ioctl(self.fd,FREE,BOFree(handle=self.handle))
            finally:self.handle=0


def finite_fp16(values):
    """Check FP16 exponent bits without the slow half-precision ufunc path."""
    values=np.asarray(values,dtype='<f2')
    return not bool(np.any((values.view('<u2') & 0x7c00)==0x7c00))


class InputTransport:
    """Reusable compiler-layout storage, including deterministic zero padding."""
    def __init__(self,descriptor):
        self.shape,self.strides,self.size=geometry(descriptor)
        self.data=bytearray(self.size)
        self.array=np.ndarray(self.shape,dtype='<f2',buffer=self.data,strides=self.strides)

    def write(self,values,buffer):
        values=np.asarray(values,dtype='<f2')
        require(values.size==np.prod(self.shape) and finite_fp16(values),'Wrong-size or nonfinite real tensor')
        self.array[...]=values.reshape(self.shape)
        buffer.write(memoryview(self.data))

    def reset(self,buffer):
        self.array.fill(0)
        buffer.write(memoryview(self.data))

    def validate_rows(self,values,offset,count):
        batches,channels,depth,height,width=self.shape
        require(0<=offset and count>0 and offset+count<=height,'Invalid cache update extent')
        values=np.asarray(values,dtype='<f2')
        require(values.size==batches*channels*depth*count*width and finite_fp16(values),
                'Wrong-size or nonfinite cache update')
        return values

    def write_rows(self,values,offset,count,buffer):
        values=self.validate_rows(values,offset,count)
        batches,channels,depth,height,width=self.shape
        self.array[:,:,:,offset:offset+count,:]=values.reshape(batches,channels,depth,count,width)
        for batch in range(batches):
            for channel in range(channels):
                for plane in range(depth):
                    start=batch*self.strides[0]+channel*self.strides[1]+plane*self.strides[2]+offset*self.strides[3]
                    end=start+(count-1)*self.strides[3]+width*2
                    buffer.write_at(memoryview(self.data)[start:end],start)


class CacheOutput:
    """Read a committed prefix before this program's next submission."""
    def __init__(self,program,tensor):
        self.program,self.tensor,self.epoch=program,tensor,program.epoch
    def read_prefix(self,count):
        program,tensor=self.program,self.tensor
        require(program.epoch==self.epoch and tensor['bank'] in program.buffers,'Stale ANE cache output')
        shape,strides,size=geometry(tensor['compiler_geometry'])
        require(0<count<=shape[3],'Invalid output cache prefix')
        mapping=program.buffers[tensor['bank']].map
        raw=getattr(mapping,'mapping',mapping)
        view=np.ndarray(shape,dtype='<f2',buffer=raw,strides=strides)
        # Copy only useful rows from every layer/head, then release the mapping
        # view. Output arrays never retain a buffer export beyond this call.
        result=view[:,:,:,:count,:].copy()
        require(finite_fp16(result),'Unwritten/nonfinite cache prefix')
        logical=list(program.shapes[tensor['name']]);logical[-2]=count
        return result.reshape(logical)


class Program:
    def __init__(self,fd,root,logical_shapes,transport='resident',head_readback='mapped',
                 verify_readback='all',kv_readback='full'):
        self.fd,self.root,self.shapes=fd,Path(root),logical_shapes
        require(transport in ('reference','packed','resident'),'Unknown input transport')
        require(head_readback in ('full','rows','mapped','native'),'Unknown head readback')
        self.head_readback=head_readback
        require(verify_readback in ('all','accepted-prefix'),'Unknown verification readback')
        require(kv_readback in ('full','prefix'),'Unknown cache output readback')
        self.verify_readback,self.kv_readback,self.epoch=verify_readback,kv_readback,0
        self.native_vocabulary=None
        self.transport=transport;self.cache_ready=False;self.cache_end=0
        self.meta=plan(root);self.buffers={};self.bootstrap=None
        self.inputs={t['name']:InputTransport(t['compiler_geometry']) for t in self.meta['io'] if t['direction']=='inputs'}
        self.input_banks={t['name']:t['bank'] for t in self.meta['io'] if t['direction']=='inputs'}
        self.poison={t['bank']:np.full(t['allocation_bytes']//2,np.nan,dtype='<f2').tobytes()
                     for t in self.meta['io'] if t['direction']=='outputs'}
        self.scratch={b['bank']:bytes(b['bytes']) for b in self.meta['buffers'] if b['role']=='scratch'}
        data=(self.root/'hwx/model.hwx').read_bytes()
        m=self.meta
        try:
            # Retain all __TEXT bytes. BAR1 points into its constants tail.
            text=bounds(data,m['text_file_offset'],m['text_file_bytes'])
            self.buffers[0]=Buffer(fd,len(text)+1);self.buffers[0].write(text)
            for b in m['buffers']:
                buffer=self.buffers[b['bank']]=Buffer(fd,b['bytes'])
                buffer.write(bounds(data,b['file_offset'],b['file_bytes']) if b['file_bytes'] else bytes(b['bytes']))
            self.bootstrap=Buffer(fd,m['td_size'])
            bootstrap=bytearray(text[:m['td_size']])
            first=struct.unpack_from('<I',bootstrap)[0]
            struct.pack_into('<I',bootstrap,0,(first&~(255<<16))|(64<<16))
            self.bootstrap.write(bootstrap)
        except BaseException:self.close();raise
    def reset_transport(self):
        self.epoch=getattr(self,'epoch',0)+1
        self.cache_ready=False;self.cache_end=0
        if self.transport=='resident' and {'cache_k','cache_v'}<=self.inputs.keys():
            for name in ('cache_k','cache_v'):
                self.inputs[name].reset(self.buffers[self.input_banks[name]])
            self.cache_ready=True
    def commit_cache(self,k,v,offset,count):
        if not self.cache_ready:return
        require(offset==self.cache_end,'Cache update must append the accepted prefix')
        # Validate both updates before changing either resident surface.
        for name,values in (('cache_k',k),('cache_v',v)):
            self.inputs[name].validate_rows(values,offset,count)
        for name,values in (('cache_k',k),('cache_v',v)):
            self.inputs[name].write_rows(values,offset,count,self.buffers[self.input_banks[name]])
        self.cache_end+=count
    def submit(self,inputs):
        self.epoch+=1
        for b in self.meta['buffers']:
            if b['role']=='scratch': self.buffers[b['bank']].write(self.scratch[b['bank']])
        for t in self.meta['io']:
            if t['direction']=='inputs':
                name=t['name']
                if self.cache_ready and name in ('cache_k','cache_v'):
                    require(np.asarray(inputs[name]).size==np.prod(self.inputs[name].shape),'Wrong-size resident cache')
                elif self.transport=='reference':
                    self.buffers[t['bank']].write(pack_tensor(inputs[name],t['compiler_geometry']))
                else:self.inputs[name].write(inputs[name],self.buffers[t['bank']])
            else:self.buffers[t['bank']].write(self.poison[t['bank']])
        request=Submit(tsk_size=self.meta['tsk_size'],td_count=self.meta['td_count'],
                       td_size=self.meta['td_size'],btsp_handle=self.bootstrap.handle)
        for bank,buffer in self.buffers.items():request.handles[bank]=buffer.handle
        require(request.handles[1]==0,'BAR1 is supplied by the driver')
        ioctl(self.fd,SUBMIT,request)

    def read_outputs(self,full=False):
        outputs={}
        for t in self.meta['io']:
            if t['direction']!='outputs':continue
            if self.kv_readback=='prefix' and not full and t['name'] in ('new_k','new_v'):
                outputs[t['name']]=CacheOutput(self,t)
                continue
            raw=self.buffers[t['bank']].map[:t['allocation_bytes']]
            value=unpack_tensor(raw,t['compiler_geometry'],self.shapes[t['name']])
            require(finite_fp16(value),f'Unwritten/nonfinite {t["name"]}')
            outputs[t['name']]=value
        return outputs

    def predict(self,inputs,role=None):
        self.submit(inputs)
        return self.read_outputs()

    def read_token_ids(self,start,end,outputs=None):
        """Reduce useful vocabulary rows; keep the lowest token ID on ties."""
        tensors=[t for t in self.meta['io'] if t['direction']=='outputs']
        require(tensors and {t['name'] for t in tensors}=={f'logits{i}' for i in range(len(tensors))},
                'Token readback requires only numbered vocabulary outputs')
        tensors.sort(key=lambda t:int(t['name'][6:]))
        if outputs is None and self.head_readback=='native':
            from m1_native_transport import Vocabulary
            if self.native_vocabulary is None:
                surfaces=[]
                for tensor in tensors:
                    shape,strides,size=geometry(tensor['compiler_geometry'])
                    require(shape[:3]==(1,1,1) and 0<=start<end<=shape[3],
                            'Unsupported vocabulary layout or requested rows')
                    mapping=self.buffers[tensor['bank']].map
                    surfaces.append((getattr(mapping,'mapping',mapping),shape[3],shape[4],strides[3]))
                self.native_vocabulary=Vocabulary(surfaces)
            return self.native_vocabulary.ids(start,end)
        if outputs is None and self.head_readback=='full':outputs=self.read_outputs()
        best=np.full(end-start,-np.inf,np.float32);ids=np.zeros(end-start,np.int32)
        base=0
        for tensor in tensors:
            shape,strides,size=geometry(tensor['compiler_geometry'])
            batches,channels,depth,height,width=shape
            require((batches,channels,depth)==(1,1,1) and 0<=start<end<=height,
                    'Unsupported vocabulary layout or requested rows')
            if outputs is not None:
                logits=np.asarray(outputs[tensor['name']])[0,start:end].astype(np.float32)
            else:
                mapping=self.buffers[tensor['bank']].map
                if self.head_readback=='rows':
                    raw=mapping[start*strides[3]:(end-1)*strides[3]+width*2]
                    logits=np.ndarray((end-start,width),dtype='<f2',buffer=raw,
                                      strides=(strides[3],2)).astype(np.float32)
                else:
                    # Convert directly from the mapped surface, avoiding a full
                    # intermediate memcpy. Diagnostic map wrappers expose mapping.
                    raw=getattr(mapping,'mapping',mapping)
                    logits=np.ndarray((end-start,width),dtype='<f2',buffer=raw,offset=start*strides[3],
                                      strides=(strides[3],2)).astype(np.float32)
            require(np.isfinite(logits).all(),'Unwritten/nonfinite vocabulary logits')
            index=logits.argmax(-1);values=logits[np.arange(end-start),index]
            better=values>best
            ids[better]=index[better]+base;best[better]=values[better]
            base+=width
        require(np.isfinite(best).all(),'Invalid ANE vocabulary logits')
        return ids.tolist()

    def predict_token_ids(self,inputs,start,end,role=None):
        self.submit(inputs)
        return self.read_token_ids(start,end)

    def read_verified_ids(self,candidates,start,end):
        require(end-start==len(candidates)+1,'Target verification needs one bonus row')
        if self.verify_readback=='all':return self.read_token_ids(start,end)
        predictions=[]
        for row in range(start,end):
            token=self.read_token_ids(row,row+1)[0]
            predictions.append(token)
            index=row-start
            if index==len(candidates) or token!=candidates[index]:break
        return predictions

    def predict_verified_ids(self,inputs,candidates,start,end,role=None):
        self.submit(inputs)
        return self.read_verified_ids(candidates,start,end)
    def close(self):
        if self.native_vocabulary is not None:self.native_vocabulary.close()
        self.native_vocabulary=None
        buffers=list(self.buffers.values())+([self.bootstrap] if self.bootstrap else [])
        self.buffers={};self.bootstrap=None
        for b in reversed(buffers):b.close()
