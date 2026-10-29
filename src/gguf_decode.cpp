// GRIMOIRE -- host GGML block decoders. SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf_decode.hpp"
#include "b70/gguf.hpp"
#include "gguf_decode_grids.hpp"
#include <cstring>
#include <limits>

// ggml evaluates the products and additions separately. Contracting them changes bits.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

namespace b70 {
namespace {
using namespace gguf_grids;
uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t u32(const uint8_t* p) { return uint32_t(u16(p)) | (uint32_t(u16(p+2)) << 16); }
uint64_t u64(const uint8_t* p) { return uint64_t(u32(p)) | (uint64_t(u32(p+4)) << 32); }
float f32(const uint8_t* p) { uint32_t bits=u32(p);float x;std::memcpy(&x,&bits,4);return x; }
double f64(const uint8_t* p) { uint64_t bits=u64(p);double x;std::memcpy(&x,&bits,8);return x; }
float bits_f32(uint32_t bits) { float x;std::memcpy(&x,&bits,4);return x; }
int sign(uint8_t bits,int j) { return (bits & kmask_iq2xs[j]) ? -1 : 1; }
template<class T> int grid_val(T packed,int j) { return int(int8_t((packed>>(8*j))&255)); }
bool ends_with(const std::string& s,const char* suffix) {
    size_t n=std::strlen(suffix);return s.size()>=n && s.compare(s.size()-n,n,suffix)==0;
}
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
    return bits_f32(bits);
}

bool gguf_dequantize(uint32_t type,const void* src,uint64_t n,float* dst,std::string& err,
                     const std::string& tensor_name) {
    err.clear();
    auto* info=gguf_type_info(type);
    if (!info) {err="unknown GGUF tensor type id "+std::to_string(type);return false;}
    if (n%info->block_elems) {err="element count is not a whole GGUF block for "+std::string(info->name);return false;}
    if (type>=24 && type<=27 && !ends_with(tensor_name,".weight") && !ends_with(tensor_name,".bias")) {
        err="integer tensor "+(tensor_name.empty()?"<unnamed>":tensor_name)+" ("+info->name+") is not a weight or bias";
        return false;
    }
    auto* p=static_cast<const uint8_t*>(src);
    if (type==0 || type==1 || type==30 || type==24 || type==25 || type==26 || type==27 || type==28) {
        for(uint64_t i=0;i<n;++i) {
            switch(type) {
            case 0: dst[i]=f32(p+4*i);break;
            case 1: dst[i]=gguf_fp16_to_f32(u16(p+2*i));break;
            case 30: dst[i]=bits_f32(uint32_t(u16(p+2*i))<<16);break;
            case 24: dst[i]=float(int8_t(p[i]));break;
            case 25: dst[i]=float(int16_t(u16(p+2*i)));break;
            case 26: dst[i]=float(int32_t(u32(p+4*i)));break;
            case 27: dst[i]=float(int64_t(u64(p+8*i)));break;
            case 28: dst[i]=float(f64(p+8*i));break;
            }
        }
        return true;
    }
    for(uint64_t b=0;b<n/info->block_elems;++b,p+=info->block_bytes) {
        float* out=dst+b*info->block_elems;
        switch(type) {
        case 2: case 3: case 6: case 7: {
            float d=gguf_fp16_to_f32(u16(p));
            float m=(type==3 || type==7)?gguf_fp16_to_f32(u16(p+2)):0.0f;
            const uint8_t* qh=(type==6?p+2:type==7?p+4:nullptr);
            const uint8_t* qs=p+(type==2?2:type==3?4:type==6?6:8);
            uint32_t high=qh?u32(qh):0;
            for(int i=0;i<32;++i) {
                int q=(qs[i%16] >> (i/16*4))&15;
                if(qh) q|=int((high>>i)&1)<<4;
                out[i]=type==2?d*float(q-8):type==6?d*float(q-16):d*float(q)+m;
            }
            break;
        }
        case 8: case 9: {
            float d=gguf_fp16_to_f32(u16(p));
            const uint8_t* qs=p+(type==8?2:4);
            for(int i=0;i<32;++i) out[i]=d*float(int8_t(qs[i]));
            break;
        }
        case 10: {
            float d=gguf_fp16_to_f32(u16(p+80)),dm=gguf_fp16_to_f32(u16(p+82));
            for(int i=0;i<256;++i) {
                int g=i/16, q=(p[16+(i/128)*32+(i%32)] >> (2*((i%128)/32)))&3;
                float dl=d*float(p[g]&15), ml=dm*float(p[g]>>4);
                out[i]=dl*float(q)-ml;
            }
            break;
        }
        case 11: {
            float d=gguf_fp16_to_f32(u16(p+108));const uint8_t* s=p+96;
            for(int i=0;i<256;++i) {
                int g=i/16;
                int low=(s[g%8] >> (g/8*4))&15;
                int high=(s[8+g%4] >> (2*(g/4)))&3;
                int sc=(low|(high<<4))-32;
                int q=(p[32+(i/128)*32+(i%32)] >> (2*((i%128)/32)))&3;
                q-=((p[i%32]>>(i/32))&1)?0:4;
                out[i]=(d*float(sc))*float(q);
            }
            break;
        }
        case 12: case 13: {
            float d=gguf_fp16_to_f32(u16(p)),dm=gguf_fp16_to_f32(u16(p+2));
            const uint8_t* scales=p+4,*qh=type==13?p+16:nullptr,*qs=type==13?p+48:p+16;
            for(int g=0;g<8;++g) {
                int sc,m;scale_min_k4(scales,g,sc,m);float ds=d*float(sc),mins=dm*float(m);
                for(int i=0;i<32;++i) {
                    int q=(qs[(g/2)*32+i] >> ((g%2)*4))&15;
                    if(qh) q|=((qh[i]>>g)&1)<<4;
                    out[g*32+i]=ds*float(q)-mins;
                }
            }
            break;
        }
        case 14: {
            const uint8_t* ql=p,*qh=p+128;float d=gguf_fp16_to_f32(u16(p+208));
            for(int half=0;half<2;++half) for(int lane=0;lane<4;++lane) for(int i=0;i<32;++i) {
                int low=(ql[half*64+(lane%2)*32+i] >> ((lane/2)*4))&15;
                int high=(qh[half*32+i] >> (lane*2))&3;
                int index=half*128+lane*32+i;
                out[index]=(d*float(int8_t(p[192+index/16])))*float((low|(high<<4))-32);
            }
            break;
        }
        case 15: {
            float d=f32(p);
            for(int i=0;i<256;++i) out[i]=d*float(int8_t(p[4+i]));
            break;
        }
        case 16: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<8;++g) {
                const uint8_t* q=p+2+g*8;uint32_t sc=u32(q+4);
                float db=(d*(0.5f+float(sc>>28)))*0.25f;
                for(int j=0;j<4;++j) {
                    int grid=q[j],bits=ksigns_iq2xs[(sc>>(7*j))&127];
                    for(int k=0;k<8;++k) out[g*32+j*8+k]=(db*float(grid_val(iq2xxs_grid[grid],k)))*float(sign(bits,k));
                }
            }
            break;
        }
        case 17: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<16;++g) {
                int sc=(p[66+g/2]>>(4*(g%2)))&15;float db=(d*(0.5f+float(sc)))*0.25f;
                for(int j=0;j<2;++j) {
                    uint16_t q=u16(p+2+2*(g*2+j));int grid=q&511,bits=ksigns_iq2xs[q>>9];
                    for(int k=0;k<8;++k) out[g*16+j*8+k]=(db*float(grid_val(iq2xs_grid[grid],k)))*float(sign(bits,k));
                }
            }
            break;
        }
        case 22: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<16;++g) {
                int sc=(p[74+g/2]>>(4*(g%2)))&15;float db=(d*(0.5f+float(sc)))*0.25f;
                for(int j=0;j<2;++j) {
                    int i=g*2+j,grid=p[2+i]|(((p[66+i/4]>>(2*(i%4)))&3)<<8),bits=p[34+i];
                    for(int k=0;k<8;++k) out[i*8+k]=(db*float(grid_val(iq2s_grid[grid],k)))*float(sign(bits,k));
                }
            }
            break;
        }
        case 18: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<8;++g) {
                uint32_t sc=u32(p+66+4*g);float db=(d*(0.5f+float(sc>>28)))*0.5f;
                for(int j=0;j<4;++j) {
                    int bits=ksigns_iq2xs[(sc>>(7*j))&127];
                    for(int h=0;h<2;++h) {
                        int grid=p[2+g*8+j*2+h];
                        for(int k=0;k<4;++k) out[g*32+j*8+h*4+k]=(db*float(grid_val(iq3xxs_grid[grid],k)))*float(sign(bits,h*4+k));
                    }
                }
            }
            break;
        }
        case 21: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<8;++g) {
                int sc=(p[106+g/2]>>(4*(g%2)))&15;float db=d*float(1+2*sc);
                for(int j=0;j<8;++j) {
                    int i=g*8+j,grid=p[2+i]|(((p[66+i/8]>>(i%8))&1)<<8);
                    int bits=p[74+i/2];
                    for(int k=0;k<4;++k) out[i*4+k]=(db*float(grid_val(iq3s_grid[grid],k)))*float(sign(bits,(i%2)*4+k));
                }
            }
            break;
        }
        case 19: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int g=0;g<8;++g) {
                uint16_t h=u16(p+34+2*g);float dl=d*float(1+2*((h>>12)&7));
                float delta=(h&0x8000)?-0.125f:0.125f;
                for(int j=0;j<4;++j) {
                    int grid=p[2+g*4+j]|(((h>>(3*j))&7)<<8);
                    for(int k=0;k<8;++k) out[g*32+j*8+k]=dl*(float(grid_val(iq1s_grid[grid],k))+delta);
                }
            }
            break;
        }
        case 29: {
            uint16_t d_bits=0;
            for(int i=0;i<4;++i) d_bits|=((u16(p+48+i*2)>>12)&15)<<(4*i);
            float d=gguf_fp16_to_f32(d_bits);
            for(int g=0;g<16;++g) {
                int sc=(u16(p+48+2*(g/4))>>(3*(g%4)))&7;
                float dl=d*float(2*sc+1);
                uint8_t h=p[32+g];
                for(int j=0;j<2;++j) {
                    int high=(h>>(4*j))&15,grid=p[g*2+j]|((high&7)<<8);
                    float delta=(high&8)?-0.125f:0.125f;
                    for(int k=0;k<8;++k) out[g*16+j*8+k]=dl*(float(grid_val(iq1s_grid[grid],k))+delta);
                }
            }
            break;
        }
        case 20: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int i=0;i<32;++i) {
                int q=(p[2+i%16]>>(4*(i/16)))&15;
                out[i]=d*float(kvalues_iq4nl[q]);
            }
            break;
        }
        case 23: {
            float d=gguf_fp16_to_f32(u16(p)),h=u16(p+2);
            for(int g=0;g<8;++g) {
                int low=(p[4+g/2]>>(4*(g%2)))&15;
                int sc=(low|(((int(h)>>(2*g))&3)<<4))-32;
                float dl=d*float(sc);
                for(int i=0;i<32;++i) {
                    int q=(p[8+g*16+i%16]>>(4*(i/16)))&15;
                    out[g*32+i]=dl*float(kvalues_iq4nl[q]);
                }
            }
            break;
        }
        case 34: {
            float d=gguf_fp16_to_f32(u16(p+52));
            for(int i=0;i<256;++i) {
                int byte,exp;
                if(i<160) {exp=i/32;byte=p[i%32];}
                else if(i<240) {int x=i-160;exp=x/16;byte=p[32+x%16];}
                else {int x=i-240;exp=x/4;byte=p[48+x%4];}
                int v=byte;
                while(exp--) v=(v*3)&255;
                int trit=((v*3)>>8)-1;
                out[i]=d*float(trit);
            }
            break;
        }
        case 35: {
            float d=gguf_fp16_to_f32(u16(p+64));
            for(int i=0;i<256;++i) {
                int q=(p[(i/128)*32+i%32]>>(2*((i%128)/32)))&3;
                out[i]=d*float(q-1);
            }
            break;
        }
        case 39: {
            uint8_t e=p[0];uint32_t bits=e<2?(0x00200000u<<e):(uint32_t(e-1)<<23);
            float d=bits_f32(bits);static constexpr int8_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
            for(int i=0;i<32;++i) {
                int q=(p[1+i%16]>>(4*(i/16)))&15;
                out[i]=d*float(values[q]);
            }
            break;
        }
        case 40: {
            static constexpr int8_t values[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
            for(int s=0;s<4;++s) {
                uint8_t x=p[s];int exp=(x>>3)&15,man=x&7;
                // ggml_ue4m3_to_fp32 returns half the UE4M3 scale for doubled E2M1 values.
                float d=(x==0 || x==0x7f)?0.0f:
                    exp==0?float(man)*0.0009765625f:
                    (1.0f+float(man)*0.125f)*bits_f32(uint32_t(exp+120)<<23)*0.5f;
                for(int i=0;i<16;++i) {
                    int q=(p[4+s*8+i%8]>>(4*(i/8)))&15;
                    out[s*16+i]=d*float(values[q]);
                }
            }
            break;
        }
        case 41: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int i=0;i<128;++i) out[i]=((p[2+i/8]>>(i%8))&1)?d:-d;
            break;
        }
        case 42: {
            float d=gguf_fp16_to_f32(u16(p));
            for(int i=0;i<64;++i) out[i]=float(((p[2+i/4]>>(2*(i%4)))&3)-1)*d;
            break;
        }
        default: err=std::string("GGUF dequantization unsupported for ")+info->name;return false;
        }
    }
    return true;
}
} // namespace b70
