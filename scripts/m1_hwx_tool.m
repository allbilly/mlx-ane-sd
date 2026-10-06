// Export M1 HWX, or evaluate that exact file with caller-provided raw surfaces.
// Private API signatures checked against freedomtan/ane_pmu_profiler.
// No compiler, CoreML, Metal, or synthetic inputs are used in eval mode.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/message.h>
#include <dlfcn.h>
#include <string.h>
typedef int (*CompileFn)(NSDictionary *, NSDictionary *, void (^)(unsigned int, NSDictionary *));
static id loadedMILModel;
static void saveJSON(id object, NSString *path) {
    NSData *data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingPrettyPrinted error:nil];
    if (!data || ![data writeToFile:path atomically:YES]) @throw @"Cannot write receipt";
}
static int exportHWX(NSString *root, NSString *out) {
    void *lib=dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",RTLD_NOW);
    CompileFn compile=lib ? (CompileFn)dlsym(lib,"ANECCompile") : NULL;
    if (!compile) return 1;
    __block unsigned int status=UINT_MAX;
    int result=compile(@{@"InputNetworks":@[@{@"NetworkSourceFileName":@"model.mil",@"NetworkSourcePath":[root stringByAppendingString:@"/"]}],
                        @"OutputFilePath":[out stringByAppendingString:@"/"],@"OutputFileName":@"model.hwx"},
                       @{@"TargetArchitecture":@"h13g",@"DumpStatusDictionaryToFile":@YES},
                       ^(unsigned int s,NSDictionary *info){status=s;if(s)NSLog(@"%@",info);});
    BOOL ok=result==0 && status==0 && [NSFileManager.defaultManager fileExistsAtPath:[out stringByAppendingPathComponent:@"model.hwx"]];
    saveJSON(@{@"compile_result":@(result),@"compile_status":@(status),@"exported":@(ok),@"architecture":@"h13g"},
             [out stringByAppendingPathComponent:@"export.json"]);
    return ok ? 0 : 1;
}
static int evalHWX(NSString *hwx, NSString *casePath, NSString *out, BOOL mil) {
    if (!dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine",RTLD_NOW)) return 1;
    NSDictionary *desc=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:casePath] options:0 error:nil];
    if (!desc || ![desc[@"inputs"] count] || ![desc[@"outputs"] count]) return 1;
    id client=((id(*)(Class,SEL))objc_msgSend)(NSClassFromString(@"_ANEClient"),sel_registerName("sharedConnection"));
    id model=nil;
    if(mil && loadedMILModel) model=loadedMILModel;
    else if(mil) {
        NSData *text=[NSData dataWithContentsOfFile:[hwx stringByAppendingPathComponent:@"original.mil"]];
        NSDictionary *weights=@{@"@model_path/weights/weight.bin":@{@"offset":@0,@"data":[NSData dataWithContentsOfFile:[hwx stringByAppendingPathComponent:@"weights/weight.bin"]]}};
        id descriptor=((id(*)(Class,SEL,id,id,id))objc_msgSend)(NSClassFromString(@"_ANEInMemoryModelDescriptor"),sel_registerName("modelWithMILText:weights:optionsPlist:"),text,weights,nil);
        model=((id(*)(Class,SEL,id))objc_msgSend)(NSClassFromString(@"_ANEInMemoryModel"),sel_registerName("inMemoryModelWithDescriptor:"),descriptor);
        id hex=((id(*)(id,SEL))objc_msgSend)(model,sel_registerName("hexStringIdentifier"));
        NSString *tmp=[NSTemporaryDirectory() stringByAppendingPathComponent:hex];
        [NSFileManager.defaultManager createDirectoryAtPath:[tmp stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
        [text writeToFile:[tmp stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weights[@"@model_path/weights/weight.bin"][@"data"] writeToFile:[tmp stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
    } else model=((id(*)(Class,SEL,id,id))objc_msgSend)(NSClassFromString(@"_ANEModel"),sel_registerName("modelAtURL:key:"),[NSURL fileURLWithPath:hwx],@"main");
    NSError *error=nil;
    BOOL loaded=NO;
    if(mil && loadedMILModel) loaded=YES;
    else if(mil && model) {
        BOOL compiled=((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("compileWithQoS:options:error:"),21,@{},&error);
        loaded=compiled && ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(model,sel_registerName("loadWithQoS:options:error:"),21,@{},&error);
        if(loaded) {
            loadedMILModel=model;
            id program=((id(*)(id,SEL))objc_msgSend)(model,sel_registerName("program"));
            NSLog(@"program class=%@ %@",NSStringFromClass([program class]),program);
        }
    } else loaded=model && ((BOOL(*)(id,SEL,id,id,unsigned int,NSError**))objc_msgSend)(client,sel_registerName("loadModel:options:qos:error:"),model,@{@"kANEFModelType":@"kANEFModelPreCompiled"},21,&error);
    if (!loaded) { NSLog(@"HWX load failed: %@",error);return 1; }
    NSMutableArray *in=[NSMutableArray array],*ix=[NSMutableArray array],*ot=[NSMutableArray array],*ox=[NSMutableArray array],*surfaces=[NSMutableArray array],*bindings=[NSMutableArray array];
    BOOL ok=YES;
    for(NSString *direction in @[@"inputs",@"outputs"]) {
        NSArray *tensors=desc[direction];
        for(NSUInteger index=0;index<tensors.count;index++) {
            NSDictionary *tensor=tensors[index];
            size_t bytes=[tensor[@"allocation_bytes"] unsignedLongLongValue];
            if(!bytes || bytes>512*1024*1024) { ok=NO;break; }
            IOSurfaceRef surface=IOSurfaceCreate((CFDictionaryRef)@{(id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,(id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),(id)kIOSurfaceAllocSize:@(bytes)});
            if(!surface) { ok=NO;break; }
            [surfaces addObject:CFBridgingRelease(surface)];
            IOSurfaceLock(surface,0,NULL);
            memset(IOSurfaceGetBaseAddress(surface),0,IOSurfaceGetAllocSize(surface));
            if([direction isEqual:@"inputs"]) {
                NSData *data=[NSData dataWithContentsOfFile:[[casePath stringByDeletingLastPathComponent] stringByAppendingPathComponent:tensor[@"file"]]];
                if(data.length!=bytes) ok=NO;
                else memcpy(IOSurfaceGetBaseAddress(surface),data.bytes,bytes);
            }
            IOSurfaceUnlock(surface,0,NULL);
            id wrapped=((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(NSClassFromString(@"_ANEIOSurfaceObject"),sel_registerName("objectWithIOSurface:"),surface);
            if(!wrapped) { ok=NO;break; }
            BOOL input=[direction isEqual:@"inputs"];
            [(input ? in:ot) addObject:wrapped];[(input ? ix:ox) addObject:@(index)];
            [bindings addObject:@{@"direction":direction,@"index":@(index),@"name":tensor[@"name"],@"allocation_bytes":@(IOSurfaceGetAllocSize(surface)),@"logical_surface_bytes":@(bytes),@"iosurface_id":@(IOSurfaceGetID(surface)),@"device_iova":NSNull.null}];
        }
        if(!ok) break;
    }
    double elapsed=0;
    if(ok) {
        id request=((id(*)(Class,SEL,id,id,id,id,id,id))objc_msgSend)(NSClassFromString(@"_ANERequest"),sel_registerName("requestWithInputs:inputIndices:outputs:outputIndices:perfStats:procedureIndex:"),in,ix,ot,ox,nil,@0);
        double begin=NSDate.timeIntervalSinceReferenceDate;
        if(mil) ok=request && ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(model,sel_registerName("evaluateWithQoS:options:request:error:"),21,@{},request,&error);
        else ok=request && ((BOOL(*)(id,SEL,id,id,id,unsigned int,NSError**))objc_msgSend)(client,sel_registerName("evaluateWithModel:options:request:qos:error:"),model,@{},request,21,&error);
        elapsed=NSDate.timeIntervalSinceReferenceDate-begin;
        if(ok) for(NSUInteger i=0;i<ot.count;i++) {
            IOSurfaceRef surface=(__bridge IOSurfaceRef)surfaces[in.count+i];
            size_t bytes=[desc[@"outputs"][i][@"allocation_bytes"] unsignedLongLongValue];
            IOSurfaceLock(surface,kIOSurfaceLockReadOnly,NULL);
            [[NSData dataWithBytes:IOSurfaceGetBaseAddress(surface) length:bytes] writeToFile:[out stringByAppendingPathComponent:[NSString stringWithFormat:@"output-%lu.bin",(unsigned long)i]] atomically:YES];
            IOSurfaceUnlock(surface,kIOSurfaceLockReadOnly,NULL);
        }
    }
    saveJSON(@{@"runtime":mil ? @"_ANEInMemoryModel MIL" : @"_ANEClient precompiled HWX",@"evaluated":@(ok),@"seconds":@(elapsed),@"surfaces":bindings,@"error":error ? error.description : @"",@"live_mmio_captured":@NO},[out stringByAppendingPathComponent:@"eval.json"]);
    // The process exits after this single case, releasing model and surfaces.
    return ok ? 0 : 1;
}
int main(int argc,char **argv) { @autoreleasepool {
    if(argc<4) return 2;
    if(argc==4 && !strcmp(argv[1],"eval-mil-batch")) {
        NSArray *cases=[NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:@(argv[3])] options:0 error:nil];
        if(!cases.count) return 2;
        for(NSDictionary *entry in cases) { @autoreleasepool {
            NSString *dest=entry[@"output"];
            if([NSFileManager.defaultManager fileExistsAtPath:dest]) return 2;
            if(![NSFileManager.defaultManager createDirectoryAtPath:dest withIntermediateDirectories:YES attributes:nil error:nil]) return 1;
            int result=evalHWX(@(argv[2]),entry[@"case"],dest,YES);
            if(result) return result;
        }}
        return 0;
    }
    NSString *out=[@(argv[argc-1]) stringByStandardizingPath];
    if([NSFileManager.defaultManager fileExistsAtPath:out]) return 2;
    if(![NSFileManager.defaultManager createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil]) return 1;
    if(argc==4 && !strcmp(argv[1],"export")) return exportHWX(@(argv[2]),out);
    if(argc==5 && !strcmp(argv[1],"eval")) return evalHWX(@(argv[2]),@(argv[3]),out,NO);
    if(argc==5 && !strcmp(argv[1],"eval-mil")) return evalHWX(@(argv[2]),@(argv[3]),out,YES);
    return 2;
}}
