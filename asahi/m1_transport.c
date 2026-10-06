/* Exact FP16 vocabulary reduction from ANE mappings, using wide NEON reads.
 * No fast-math: finite values, signed-zero ties and lowest-index ties follow
 * the FP32 NumPy reference. This changes CPU transport only, never ANE tasks.
 */
#include <arm_neon.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

static inline void update(uint16x8_t bits,uint32_t col,uint16x8_t *best,
                          uint16x8_t *ids,uint16x8_t *bad) {
    const uint16x8_t lanes={0,1,2,3,4,5,6,7};
    *bad=vorrq_u16(*bad,vceqq_u16(vandq_u16(bits,vdupq_n_u16(0x7c00)),vdupq_n_u16(0x7c00)));
    uint16x8_t negative=vcltq_s16(vreinterpretq_s16_u16(bits),vdupq_n_s16(0));
    uint16x8_t keys=veorq_u16(bits,vbslq_u16(negative,vdupq_n_u16(0xffff),vdupq_n_u16(0x8000)));
    uint16x8_t zero=vceqq_u16(vandq_u16(bits,vdupq_n_u16(0x7fff)),vdupq_n_u16(0));
    keys=vbslq_u16(zero,vdupq_n_u16(0x8000),keys);
    *ids=vbslq_u16(vcgtq_u16(keys,*best),vaddq_u16(lanes,vdupq_n_u16(col)),*ids);
    *best=vmaxq_u16(keys,*best);
}

static int reduce_row(const uint16_t *src, uint32_t width, int32_t *index, float *maximum) {
    uint32_t winner=0;
    if (width<=UINT16_MAX) {
        uint16x8_t best[4]={vdupq_n_u16(0),vdupq_n_u16(0),vdupq_n_u16(0),vdupq_n_u16(0)};
        uint16x8_t ids[4]={vdupq_n_u16(0),vdupq_n_u16(0),vdupq_n_u16(0),vdupq_n_u16(0)};
        uint16x8_t bad = vdupq_n_u16(0);
        uint32_t col=0;
        for (; col+32<=width; col+=32) {
            uint16x8x4_t bits=vld1q_u16_x4(src+col);
            update(bits.val[0],col,best,ids,&bad);
            update(bits.val[1],col+8,best+1,ids+1,&bad);
            update(bits.val[2],col+16,best+2,ids+2,&bad);
            update(bits.val[3],col+24,best+3,ids+3,&bad);
        }
        for (; col+8<=width; col+=8) update(vld1q_u16(src+col),col,best,ids,&bad);
        if (vmaxvq_u16(bad)) return -2;
        uint16_t key=vmaxvq_u16(vmaxq_u16(vmaxq_u16(best[0],best[1]),vmaxq_u16(best[2],best[3])));
        winner=UINT16_MAX;
        for (uint32_t part=0;part<4;++part) {
            uint32_t id=vminvq_u16(vbslq_u16(vceqq_u16(best[part],vdupq_n_u16(key)),ids[part],vdupq_n_u16(UINT16_MAX)));
            if (id<winner) winner=id;
        }
        for (; col < width; ++col) {
            uint16_t bits=src[col];
            if ((bits & 0x7c00)==0x7c00) return -2;
            uint16_t candidate=(bits & 0x7fff)==0 ? 0x8000 : ((bits & 0x8000) ? (uint16_t)~bits : bits^0x8000);
            if (candidate>key || (candidate==key && col<winner)) { key=candidate;winner=col; }
        }
    } else {
        // Wider user-provided surfaces retain a correct scalar fallback.
        float best=-INFINITY;
        for (uint32_t col=0; col<width; ++col) {
            uint16_t bits=src[col];
            if ((bits & 0x7c00)==0x7c00) return -2;
            __fp16 half;memcpy(&half,&bits,sizeof half);
            if ((float)half>best) { best=(float)half;winner=col; }
        }
    }
    __fp16 half;memcpy(&half,src+winner,sizeof half);
    *index=winner;*maximum=(float)half;
    return 0;
}

int m1_fp16_argmax(const void *data, uint32_t rows, uint32_t width,
                   uint64_t row_stride, int32_t *indices, float *maxima, uint32_t threads) {
    if (!data || !indices || !maxima || !rows || !width ||
        row_stride < (uint64_t)width * 2 || row_stride % 2 || !threads || threads>32) return -1;
    int invalid=0;
    #pragma omp parallel for if(rows>=4 && threads>1) num_threads(threads) reduction(|:invalid)
    for (uint32_t row=0; row<rows; ++row) {
        const uint16_t *src=(const uint16_t *)((const char *)data+row*row_stride);
        if (reduce_row(src,width,indices+row,maxima+row)) invalid=1;
    }
    return invalid ? -2 : 0;
}
/* One parallel region spans all vocabulary chunks. Launching and joining a
 * worker team separately for each small chunk can outweigh the read savings. */
int m1_fp16_argmax_chunks(const uintptr_t *data, const uint64_t *strides,
                          const uint32_t *widths, uint32_t chunks, uint32_t start,
                          uint32_t rows, int32_t *indices, float *maxima, uint32_t threads) {
    if (!data || !strides || !widths || !chunks || !rows || !indices || !maxima ||
        !threads || threads>32) return -1;
    int invalid=0;
    #pragma omp parallel for if(rows>=4 && threads>1) num_threads(threads) reduction(|:invalid)
    for (uint32_t row=0; row<rows; ++row) {
        float best=-INFINITY;
        int32_t best_id=0;
        uint32_t base=0;
        for (uint32_t chunk=0; chunk<chunks; ++chunk) {
            int32_t id;
            float maximum;
            const uint16_t *source=(const uint16_t *)((const char *)data[chunk]+(start+row)*strides[chunk]);
            if (reduce_row(source,widths[chunk],&id,&maximum)) {
                invalid=1;
                break;
            }
            if (maximum>best) { best=maximum; best_id=base+id; }
            base+=widths[chunk];
        }
        indices[row]=best_id;
        maxima[row]=best;
    }
    return invalid ? -2 : 0;
}
