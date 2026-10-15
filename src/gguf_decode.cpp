// GRIMOIRE -- host GGML block decoders. SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf_decode.hpp"
#include "b70/gguf.hpp"
#include <cstring>

// Reference decoders must be bit-exact with ggml's dequantize_row_* (tests/gguf golden files).
// A fused multiply-add of `ds*q - mins` rounds once instead of twice and silently breaks that,
// and icpx/clang contract by default at -O2+ on some targets. Forbid contraction here.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace b70 {
namespace {
uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
float f32(const uint8_t* p) { uint32_t u=uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24); float f;std::memcpy(&f,&u,4);return f; }
void scale_min_k4(const uint8_t* s,int j,int& sc,int& m) {
    if (j<4) { sc=s[j]&63; m=s[j+4]&63; }
    else { sc=(s[j+4]&15)|((s[j-4]>>2)&48); m=(s[j+4]>>4)|((s[j]>>2)&48); }
}
}
float gguf_fp16_to_f32(uint16_t h) {
    uint32_t sign=uint32_t(h&0x8000)<<16;
    uint32_t exp=(h>>10)&31, mant=h&1023, bits;
    if (exp==0) {
        if (!mant) bits=sign;
        else { int e=-14; while (!(mant&1024)) {mant<<=1;--e;} mant&=1023;bits=sign|(uint32_t(e+127)<<23)|(mant<<13); }
    } else if (exp==31) bits=sign|0x7f800000u|(mant<<13);
    else bits=sign|((exp+112)<<23)|(mant<<13);
    float f;std::memcpy(&f,&bits,4);return f;
}

bool gguf_dequantize(uint32_t type,const void* src,uint64_t n,float* dst,std::string& err) {
    err.clear();
    auto* info=gguf_type_info(type);
    if (!info) {err="unknown GGUF tensor type id "+std::to_string(type);return false;}
    if (type!=0 && type!=1 && type!=30 && type!=8 && type!=12 && type!=13 && type!=14) {
        err=std::string("GGUF dequantization unsupported for ")+info->name;return false;
    }
    if (n%info->block_elems) {err="element count is not a whole GGUF block";return false;}
    auto* p=static_cast<const uint8_t*>(src);
    if (type==0) { for(uint64_t i=0;i<n;++i) dst[i]=f32(p+4*i);return true; }
    if (type==1 || type==30) {
        for(uint64_t i=0;i<n;++i) {
            uint16_t x=u16(p+2*i);
            if(type==1) dst[i]=gguf_fp16_to_f32(x);
            else { uint32_t bits=uint32_t(x)<<16;std::memcpy(dst+i,&bits,4); }
        }
        return true;
    }
    for(uint64_t b=0;b<n/info->block_elems;++b,p+=info->block_bytes) {
        float* out=dst+b*info->block_elems;
        if(type==8) {
            float d=gguf_fp16_to_f32(u16(p));
            for(int i=0;i<32;++i) out[i]=d*float(int8_t(p[2+i]));
        } else if(type==12 || type==13) {
            float d=gguf_fp16_to_f32(u16(p)), dm=gguf_fp16_to_f32(u16(p+2));
            const uint8_t* scales=p+4;
            const uint8_t* qh=type==13?p+16:nullptr;
            const uint8_t* qs=type==13?p+48:p+16;
            for(int group=0;group<8;++group) {
                int sc,m;scale_min_k4(scales,group,sc,m);
                float ds=d*float(sc), mins=dm*float(m);
                for(int i=0;i<32;++i) {
                    int q=(qs[(group/2)*32+i] >> ((group%2)*4))&15;
                    if(qh) q|=((qh[i]>>group)&1)<<4;
                    out[group*32+i]=ds*float(q)-mins;
                }
            }
        } else { // Q6_K: four 64-element groups; low nibbles and high pairs interleave.
            const uint8_t* ql=p;const uint8_t* qh=p+128;
            const int8_t* scales=reinterpret_cast<const int8_t*>(p+192);
            float d=gguf_fp16_to_f32(u16(p+208));
            for(int half=0;half<2;++half) for(int lane=0;lane<4;++lane)
                for(int i=0;i<32;++i) {
                    int low=(ql[half*64+(lane%2)*32+i] >> ((lane/2)*4))&15;
                    int high=(qh[half*32+i] >> (lane*2))&3;
                    int index=half*128+lane*32+i;
                    int q=(low|(high<<4))-32;
                    out[index]=(d*float(scales[index/16]))*float(q);
                }
        }
    }
    return true;
}
} // namespace b70
