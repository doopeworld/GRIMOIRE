// Standalone C reference for GGML scalar layouts, Q8_1, Q8_K, Q1_0, Q2_0.
// Q8_K, Q1_0 and Q2_0 loops below are direct copies of the arithmetic in
// pinned ggml-quants.c dequantize_row_*; scalar casts and Q8_1 lack upstream
// dequantize_row_* functions and follow their pinned ggml-common.h layouts.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t u32(const uint8_t *p) { return (uint32_t)u16(p) | ((uint32_t)u16(p+2)<<16); }
static uint64_t u64(const uint8_t *p) { return (uint64_t)u32(p) | ((uint64_t)u32(p+4)<<32); }
static float fp16(uint16_t h) {
    uint32_t s=(uint32_t)(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,b;
    if (!e) { if (!m) b=s; else { int x=-14;while (!(m&1024)) {m<<=1;--x;}b=s|((uint32_t)(x+127)<<23)|((m&1023)<<13); } }
    else if(e==31) b=s|0x7f800000|(m<<13);
    else b=s|((e+112)<<23)|(m<<13);
    float f;memcpy(&f,&b,4);return f;
}
// Struct definitions and the three functions below are copied from
// ggml-common.h / ggml-quants.c at llama.cpp 711780ffc75b.
#define GGML_RESTRICT restrict
#define GGML_FP16_TO_FP32 fp16
#define QK1_0 128
#define QK2_0 64
#define QK_K 256
typedef struct { uint16_t d; uint8_t qs[QK1_0 / 8]; } block_q1_0;
typedef struct { uint16_t d; uint8_t qs[QK2_0 / 4]; } block_q2_0;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK1_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float neg_d = -d;

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 8;
            const int bit_offset = j % 8;
            const uint8_t bit = (x[i].qs[byte_index] >> bit_offset) & 1;
            y[i*qk + j] = bit ? d : neg_d;
        }
    }
}

void dequantize_row_q2_0(const block_q2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK2_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            const int byte_index = j / 4;
            const int bit_offset = (j % 4) * 2;
            const uint8_t q = (x[i].qs[byte_index] >> bit_offset) & 0x03;
            // 00=-1, 01=0, 10=+1, 11=+2
            y[i*qk + j] = ((int)q - 1) * d;
        }
    }
}

void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < QK_K; ++j) {
            *y++ = x[i].d * x[i].qs[j];
        }
    }
}

int main(int argc,char **argv) {
    if(argc!=3) return 2;
    int type=atoi(argv[1]);size_t n=(size_t)strtoul(argv[2],0,10);
    size_t width=type==9?36:type==15?292:type==41||type==42?18:type==24?1:type==25?2:type==26?4:8;
    size_t bytes=type==9?(n/32)*width:type==15?(n/256)*width:type==41?(n/128)*width:type==42?(n/64)*width:n*width;
    uint8_t *raw=(uint8_t*)malloc(bytes);float *out=(float*)malloc(n*sizeof(float));
    if(!raw||!out||fread(raw,1,bytes,stdin)!=bytes) return 3;
    if(type==15) dequantize_row_q8_K((const block_q8_K *)raw,out,(int64_t)n);
    else if(type==41) dequantize_row_q1_0((const block_q1_0 *)raw,out,(int64_t)n);
    else if(type==42) dequantize_row_q2_0((const block_q2_0 *)raw,out,(int64_t)n);
    else for(size_t i=0;i<n;++i) {
        uint64_t x;
        switch(type) {
        case 9: out[i]=fp16(u16(raw+(i/32)*36))*(float)(int8_t)raw[(i/32)*36+4+i%32];break;
        case 24: out[i]=(float)(int8_t)raw[i];break;
        case 25: out[i]=(float)(int16_t)u16(raw+i*2);break;
        case 26: out[i]=(float)(int32_t)u32(raw+i*4);break;
        case 27: x=u64(raw+i*8);out[i]=(float)(int64_t)x;break;
        case 28: x=u64(raw+i*8);{double d;memcpy(&d,&x,8);out[i]=(float)d;}break;
        default:return 4;
        }
    }
    if(fwrite(out,4,n,stdout)!=n) return 5;
    free(raw);free(out);return 0;
}
