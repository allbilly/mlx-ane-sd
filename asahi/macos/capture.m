// Capture evaluated runtime artifacts and a separately compiled H13G HWX.
#import <Foundation/Foundation.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*CompileFn)(NSDictionary *, NSDictionary *, void (^)(unsigned int, NSDictionary *));
static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}
static int compare_double(const void *a, const void *b) {
    double x=*(const double*)a, y=*(const double*)b;
    return (x>y)-(x<y);
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc!=2) { fprintf(stderr,"usage: capture CASE_DIRECTORY\n"); return 2; }
    NSString *root=[@(argv[1]) stringByStandardizingPath];
    NSFileManager *fm=NSFileManager.defaultManager;
    NSData *raw=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:@"case.json"]];
    NSDictionary *desc=raw ? [NSJSONSerialization JSONObjectWithData:raw options:0 error:nil] : nil;
    NSString *mil=[NSString stringWithContentsOfFile:[root stringByAppendingPathComponent:@"model.mil"] encoding:NSUTF8StringEncoding error:nil];
    if (!desc || !mil) return 1;
    NSString *out=[root stringByAppendingPathComponent:@"capture"];
    if ([fm fileExistsAtPath:out]) { fprintf(stderr,"capture directory already exists\n"); return 2; }
    [fm createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil];
    NSMutableDictionary *weights=[NSMutableDictionary dictionary];
    for (NSString *path in desc[@"blobs"]) {
        NSData *data=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:path]];
        if (!data) return 1;
        weights[[@"@model_path/" stringByAppendingString:path]]=@{@"offset":@0,@"data":data};
    }
    if (!orion_ane_init()) return 1;
    OrionProgram *program=orion_compile_mil(mil.UTF8String,weights,desc[@"label"] ? [desc[@"label"] UTF8String] : "sd-capture");
    if (!program) return 1;
    NSArray *in_desc=desc[@"inputs"], *out_desc=desc[@"outputs"];
    int ni=(int)in_desc.count, no=(int)out_desc.count;
    IOSurfaceRef inputs[ni], outputs[no];
    memset(inputs,0,sizeof(inputs)); memset(outputs,0,sizeof(outputs));
    bool ok=ni>0 && no>0;
    for (int i=0;i<ni && ok;i++) {
        NSArray *shape=in_desc[i][@"shape"];
        int channels=[shape[1] intValue], seq=[shape[3] intValue];
        NSData *data=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:in_desc[i][@"file"]]];
        inputs[i]=orion_tensor_create(channels,seq);
        ok=inputs[i] && data.length==(size_t)channels*seq*2;
        if (ok) orion_tensor_write(inputs[i],data.bytes,data.length);
    }
    for (int i=0;i<no && ok;i++) {
        NSArray *shape=out_desc[i][@"shape"];
        outputs[i]=orion_tensor_create([shape[1] intValue],[shape[3] intValue]);
        ok=outputs[i]!=NULL;
    }
    double samples[20];
    for (int i=0;i<22 && ok;i++) {
        double start=now_ms();
        ok=orion_eval(program,inputs,ni,outputs,no);
        if (i>=2) samples[i-2]=now_ms()-start;
    }
    NSMutableArray *checks=[NSMutableArray array];
    for (int i=0;i<no && ok;i++) {
        NSArray *shape=out_desc[i][@"shape"];
        size_t count=(size_t)[shape[1] intValue]*[shape[3] intValue];
        _Float16 *values=malloc(count*2);
        orion_tensor_read(outputs[i],values,count*2);
        NSData *reference=[NSData dataWithContentsOfFile:[root stringByAppendingPathComponent:out_desc[i][@"reference"]]];
        if (!reference || reference.length!=count*2) { free(values); ok=false; break; }
        const _Float16 *expected=reference.bytes;
        double squared=0,denom=0,max_error=0;
        for (size_t j=0;j<count;j++) {
            if (!isfinite((float)values[j])) { ok=false; break; }
            double delta=(float)values[j]-(float)expected[j];
            squared+=delta*delta; denom+=(float)expected[j]*(float)expected[j];
            max_error=fmax(max_error,fabs(delta));
        }
        NSString *file=[NSString stringWithFormat:@"output-%d.bin",i];
        [[NSData dataWithBytes:values length:count*2] writeToFile:[out stringByAppendingPathComponent:file] atomically:YES];
        [checks addObject:@{@"file":file,@"max_abs_difference_from_host":@(max_error),
                            @"relative_rmse_from_host":@(sqrt(squared/fmax(denom,1e-30)))}];
        free(values);
    }
    bool retained=orion_program_copy_artifacts(program,[[out stringByAppendingPathComponent:@"evaluated-runtime"] UTF8String]);
    for (int i=0;i<ni;i++) if (inputs[i]) CFRelease(inputs[i]);
    for (int i=0;i<no;i++) if (outputs[i]) CFRelease(outputs[i]);
    orion_release_program(program);
    if (!ok || !retained) return 1;
    qsort(samples,20,sizeof(double),compare_double);
    void *library=dlopen("/System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler",RTLD_NOW);
    CompileFn compile=library ? (CompileFn)dlsym(library,"ANECCompile") : NULL;
    NSString *hwx=[out stringByAppendingPathComponent:@"offline-hwx"];
    [fm createDirectoryAtPath:hwx withIntermediateDirectories:YES attributes:nil error:nil];
    __block unsigned int status=0;
    int result=compile ? compile(@{@"InputNetworks":@[@{@"NetworkSourceFileName":@"model.mil",@"NetworkSourcePath":[root stringByAppendingString:@"/"]}],
        @"OutputFilePath":[hwx stringByAppendingString:@"/"],@"OutputFileName":@"model.hwx"},
        @{@"TargetArchitecture":@"h13g",@"DumpStatusDictionaryToFile":@YES},
        ^(unsigned int s,NSDictionary *info){status=s;if(s)NSLog(@"HWX export: %@",info);}) : -1;
    bool exported=result==0 && status==0 && [fm fileExistsAtPath:[hwx stringByAppendingPathComponent:@"model.hwx"]];
    NSDictionary *receipt=@{@"label":desc[@"label"],@"runtime_evaluated":@YES,
        @"runtime_artifacts_retained":@(retained),@"runtime_eval_median_ms":@((samples[9]+samples[10])/2),
        @"samples":@20,@"output_checks":checks,@"offline_hwx_exported":@(exported),
        @"offline_hwx_evaluated":@NO,@"offline_compile_result":@(result),@"offline_compile_status":@(status)};
    NSData *json=[NSJSONSerialization dataWithJSONObject:receipt options:NSJSONWritingPrettyPrinted error:nil];
    [json writeToFile:[out stringByAppendingPathComponent:@"receipt.json"] atomically:YES];
    printf("%s\n",[[[NSString alloc]initWithData:json encoding:NSUTF8StringEncoding] UTF8String]);
    return exported ? 0 : 3;
}}
