// Included by runq.c after the scalar transformer helpers.
#ifndef QWEN3_ANE_PREFILL_H
#define QWEN3_ANE_PREFILL_H
#include "ane_matmul.h"

static float *prefill_storage(size_t count) {
    float *p=calloc(count,sizeof(float));
    if (!p) { fprintf(stderr,"ANE prefill allocation failed.\n"); exit(1); }
    return p;
}

static void prefill_linear(NpuMatmulKind kind, int layer, const float *in,
                            float *out, int rows, int k, int n,
                            QuantizedTensor *w, QuantizedTensor *scratch) {
    if (npu_matmul_run_batch(&g_npu,kind,layer,in,out,rows)) return;
    for (int row=0;row<rows;row++) {
        quantize(scratch,(float *)in+(size_t)row*k,k);
        matmul(out+(size_t)row*n,scratch,w,k,n);
        g_npu.cpu_ops++;
    }
}

static float *forward_ane_prompt(Transformer *t, const int *ids, int count, int start) {
    Config *c=&t->config;
    TransformerWeights *w=&t->weights;
    RunState *s=&t->state;
    int dim=c->dim, hidden=c->hidden_dim, heads=c->n_heads*c->head_dim;
    int kv=c->n_kv_heads*c->head_dim, kv_mul=c->n_heads/c->n_kv_heads;
    int width=dim>heads?dim:heads;
    if (count<1 || start<0 || count>c->seq_len-start) {
        fprintf(stderr,"Prompt exceeds the context window.\n"); exit(1);
    }
    // Reused across all prompt chunks/layers. Matrix buffers are row-major.
    float *x=prefill_storage((size_t)32*dim);
    float *norm=prefill_storage((size_t)32*width);
    float *q=prefill_storage((size_t)32*heads);
    float *k=prefill_storage((size_t)32*kv), *v=prefill_storage((size_t)32*kv);
    float *attended=prefill_storage((size_t)32*heads);
    float *projected=prefill_storage((size_t)32*dim);
    float *gate=prefill_storage((size_t)32*hidden), *up=prefill_storage((size_t)32*hidden);
    float *angles=prefill_storage((size_t)32*c->head_dim);
    for (int base=0;base<count;base+=32) {
        int rows=count-base;
        if (rows>32) rows=32;
        // A short tail costs less on CPU than another fixed-width ANE batch.
        if (rows<16 && !ane_decode_enabled(&g_npu)) {
            forward_serial_prompt(t,ids+base,rows,start+base);
            break;
        }
        for (int row=0;row<rows;row++) {
            int token=ids[base+row], pos=start+base+row;
            if (token<0 || token>=c->vocab_size) { fprintf(stderr,"Invalid prompt token.\n"); exit(1); }
            memcpy(x+(size_t)row*dim,w->token_embedding_table+(size_t)token*dim,dim*sizeof(float));
            float *cs=angles+(size_t)row*c->head_dim, *sn=cs+c->head_dim/2;
            for (int j=0;j<c->head_dim/2;j++) {
                float angle=pos*powf(1e6,-(float)j/(c->head_dim/2));
                cs[j]=cosf(angle); sn[j]=sinf(angle);
            }
        }
        for (int layer=0;layer<c->n_layers;layer++) {
            for (int row=0;row<rows;row++)
                rmsnorm(norm+(size_t)row*dim,x+(size_t)row*dim,w->rms_att_weight+(size_t)layer*dim,dim);
            prefill_linear(NPU_MATMUL_WQ,layer,norm,q,rows,dim,heads,w->wq+layer,&s->xq);
            prefill_linear(NPU_MATMUL_WK,layer,norm,k,rows,dim,kv,w->wk+layer,&s->xq);
            prefill_linear(NPU_MATMUL_WV,layer,norm,v,rows,dim,kv,w->wv+layer,&s->xq);
            uint64_t layer_offset=(uint64_t)layer*c->seq_len*kv;
            for (int row=0;row<rows;row++) {
                int pos=start+base+row;
                float *cs=angles+(size_t)row*c->head_dim, *sn=cs+c->head_dim/2;
                for (int kind=0;kind<2;kind++) {
                    int nheads=kind?c->n_kv_heads:c->n_heads;
                    float *a=(kind?k:q)+(size_t)row*(kind?kv:heads);
                    float *weight=(kind?w->k_norm_weights:w->q_norm_weights)+(size_t)layer*c->head_dim;
                    for (int h=0;h<nheads;h++) {
                        float *head=a+(size_t)h*c->head_dim;
                        rmsnorm(head,head,weight,c->head_dim);
                        for (int j=0;j<c->head_dim/2;j++) {
                            float real=head[j], imag=head[j+c->head_dim/2];
                            head[j]=real*cs[j]-imag*sn[j];
                            head[j+c->head_dim/2]=real*sn[j]+imag*cs[j];
                        }
                    }
                }
                memcpy(s->key_cache+layer_offset+(size_t)pos*kv,k+(size_t)row*kv,kv*sizeof(float));
                memcpy(s->value_cache+layer_offset+(size_t)pos*kv,v+(size_t)row*kv,kv*sizeof(float));
            }
            // Each row sees only its existing cache and earlier rows in this chunk.
            for (int row=0;row<rows;row++) {
                int pos=start+base+row;
                #pragma omp parallel for
                for (int h=0;h<c->n_heads;h++) {
                    float *query=q+(size_t)row*heads+(size_t)h*c->head_dim;
                    float *scores=s->att+(size_t)h*c->seq_len;
                    for (int previous=0;previous<=pos;previous++) {
                        float *key=s->key_cache+layer_offset+(size_t)previous*kv+(size_t)(h/kv_mul)*c->head_dim;
                        float score=0;
                        for (int j=0;j<c->head_dim;j++) score+=query[j]*key[j];
                        scores[previous]=score/sqrtf(c->head_dim);
                    }
                    softmax(scores,pos+1);
                    float *out=attended+(size_t)row*heads+(size_t)h*c->head_dim;
                    memset(out,0,c->head_dim*sizeof(float));
                    for (int previous=0;previous<=pos;previous++) {
                        float *value=s->value_cache+layer_offset+(size_t)previous*kv+(size_t)(h/kv_mul)*c->head_dim;
                        for (int j=0;j<c->head_dim;j++) out[j]+=scores[previous]*value[j];
                    }
                }
            }
            prefill_linear(NPU_MATMUL_WO,layer,attended,projected,rows,heads,dim,w->wo+layer,&s->xq);
            for (int i=0;i<rows*dim;i++) x[i]+=projected[i];
            for (int row=0;row<rows;row++)
                rmsnorm(norm+(size_t)row*dim,x+(size_t)row*dim,w->rms_ffn_weight+(size_t)layer*dim,dim);
            prefill_linear(NPU_MATMUL_W1,layer,norm,gate,rows,dim,hidden,w->w1+layer,&s->xq);
            prefill_linear(NPU_MATMUL_W3,layer,norm,up,rows,dim,hidden,w->w3+layer,&s->xq);
            for (int i=0;i<rows*hidden;i++) gate[i]=(gate[i]/(1+expf(-gate[i])))*up[i];
            prefill_linear(NPU_MATMUL_W2,layer,gate,projected,rows,hidden,dim,w->w2+layer,&s->hq);
            for (int i=0;i<rows*dim;i++) x[i]+=projected[i];
        }
        if (base+rows==count) {
            rmsnorm(s->x,x+(size_t)(rows-1)*dim,w->rms_final_weight,dim);
            quantize(&s->xq,s->x,dim);
            matmul(s->logits,&s->xq,w->wcls,dim,c->vocab_size);
            g_npu.cpu_ops++;
        }
    }
    free(x); free(norm); free(q); free(k); free(v); free(attended);
    free(projected); free(gate); free(up); free(angles);
    return s->logits;
}
#endif
