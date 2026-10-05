#ifndef QWEN3_NPU_MATMUL_H
#define QWEN3_NPU_MATMUL_H

#include "qwen3.h"

typedef enum {
    NPU_MATMUL_WQ = 0,
    NPU_MATMUL_WK,
    NPU_MATMUL_WV,
    NPU_MATMUL_WO,
    NPU_MATMUL_W1,
    NPU_MATMUL_W2,
    NPU_MATMUL_W3,
    NPU_MATMUL_WCLS
} NpuMatmulKind;

#if defined(QWEN3_USE_ANE)
typedef struct {
    int enabled;
    unsigned long long npu_ops;
    unsigned long long cpu_ops;
    void *impl;
} NpuMatmulContext;
#elif defined(QWEN3_DISABLE_NPU)
typedef struct {
    int enabled;
    unsigned long long npu_ops;
    unsigned long long cpu_ops;
} NpuMatmulContext;
#else
typedef struct {
    int enabled;
    int verbose;
    unsigned long long npu_ops;
    unsigned long long cpu_ops;
    int dim;
    int hidden_dim;
    int n_layers;
    int all_heads_dim;
    int kv_dim;
    int vocab_size;
    __fp16 **wq;
    __fp16 **wk;
    __fp16 **wv;
    __fp16 **wo;
    __fp16 **w1;
    __fp16 **w2;
    __fp16 **w3;
    __fp16 *wcls;
    __fp16 *tmp_in_dim;
    __fp16 *tmp_in_hidden;
    __fp16 *tmp_in_all_heads;
} NpuMatmulContext;
#endif

void npu_matmul_reset_stats(NpuMatmulContext *ctx);

int npu_matmul_init(NpuMatmulContext *ctx, const Config *config,
                    const TransformerWeights *weights);
void npu_matmul_shutdown(NpuMatmulContext *ctx);
int npu_matmul_run(NpuMatmulContext *ctx, NpuMatmulKind kind, int layer,
                   const float *in, float *out);

#endif
