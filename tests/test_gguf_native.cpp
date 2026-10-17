// Host verification of the exact decoder used in SYCL kernels.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/weights.hpp"
#include "b70/gguf_decode.hpp"
#include "b70/gguf_qwen35.hpp"
#include "b70/native_model.hpp"
#include "b70/native_qk_tile.hpp"
#include "b70/native_qk_gemv.hpp"
#include "gguf/test_writer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <vector>
#include <unistd.h>

using namespace b70;
static int bad=0,checked=0;
#define CHECK(c) do { ++checked; if(!(c)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);++bad;} } while(0)
static uint32_t u32(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}

template <Fmt F> static void tile16(const uint8_t* row,int k,uint32_t* dst,int ld,int n){
    native_qk_tile_16_vnni<F>(row,k,dst,ld,n);
}
static void tile16(Fmt fmt,const uint8_t* row,int k,uint32_t* dst,int ld,int n){
    switch(fmt){
    case Fmt::Q4_K: tile16<Fmt::Q4_K>(row,k,dst,ld,n);break;
    case Fmt::Q5_K: tile16<Fmt::Q5_K>(row,k,dst,ld,n);break;
    case Fmt::Q6_K: tile16<Fmt::Q6_K>(row,k,dst,ld,n);break;
    case Fmt::Q8_0: tile16<Fmt::Q8_0>(row,k,dst,ld,n);break;
    case Fmt::IQ4_XS: tile16<Fmt::IQ4_XS>(row,k,dst,ld,n);break;
    case Fmt::IQ4_NL: tile16<Fmt::IQ4_NL>(row,k,dst,ld,n);break;
    default: CHECK(false);break;
    }
}
template <Fmt F> static void tile_scaled(const uint8_t* row,int k,int block_bytes,float* values){
    float scales[16],mins[8];
    native_qk_block_scales<F>(row+(k/NativeGGUFBlock<F>::elems)*block_bytes,scales,mins);
    native_qk_tile_16_scaled<F>(row,k,scales,mins,values);
}
static void tile_scaled(Fmt fmt,const uint8_t* row,int k,int block_bytes,float* values){
    switch(fmt){
    case Fmt::Q4_K: tile_scaled<Fmt::Q4_K>(row,k,block_bytes,values);break;
    case Fmt::Q5_K: tile_scaled<Fmt::Q5_K>(row,k,block_bytes,values);break;
    case Fmt::Q6_K: tile_scaled<Fmt::Q6_K>(row,k,block_bytes,values);break;
    case Fmt::Q8_0: tile_scaled<Fmt::Q8_0>(row,k,block_bytes,values);break;
    case Fmt::IQ4_XS: tile_scaled<Fmt::IQ4_XS>(row,k,block_bytes,values);break;
    case Fmt::IQ4_NL: tile_scaled<Fmt::IQ4_NL>(row,k,block_bytes,values);break;
    default: CHECK(false);break;
    }
}
static void format_test(Fmt fmt,uint32_t type,const char* name,int block_bytes){
    std::ifstream in(std::string("tests/gguf/data/")+name+".bin",std::ios::binary);
    CHECK(bool(in));if(!in)return;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)),{});
    CHECK(file.size()>=16);if(file.size()<16)return;
    const int K=native_gguf_block_elems(fmt),N=int(u32(file.data()+8))/K;
    const uint32_t raw_size=u32(file.data()+12);
    CHECK(u32(file.data()+4)==type);
    CHECK(native_gguf_fmt(type)==fmt && native_gguf_type(fmt)==type);
    CHECK(raw_size==uint32_t(N*block_bytes));
    CHECK(file.size()==16+raw_size+size_t(N)*K*4);
    CHECK(bytes_per_row(fmt,K)==block_bytes);
    CHECK(bytes_per_row(fmt,K-1)==0);
    CHECK(scales_per_row(fmt,K)==0);
    CHECK(bits_per_elem(fmt)==double(block_bytes)*8/K);
    CHECK(std::strcmp(fmt_name(fmt),std::string(name)=="Q4_K"?"q4_k":
            std::string(name)=="Q5_K"?"q5_k":std::string(name)=="Q6_K"?"q6_k":
            std::string(name)=="Q8_0"?"q8_0":std::string(name)=="IQ4_XS"?"iq4_xs":"iq4_nl")==0);
    QuantWeight w{fmt,N,K,file.data()+16,nullptr,nullptr,block_bytes,0};
    CHECK(w.scales==nullptr && w.zeros==nullptr && w.row_scales==0);
    const QuantWeight sliced=slice_quant_rows(w,1,1);
    CHECK(sliced.N==1 && sliced.K==K && sliced.fmt==fmt);
    CHECK(sliced.payload==w.payload+block_bytes && sliced.scales==nullptr);
    for(int k=0;k<K;++k)CHECK(f32_to_bits(sliced.at(0,k))==
                            u32(file.data()+16+raw_size+(size_t(K)+k)*4));
    try {
        float dummy[256]{};
        (void)quantize(dummy,1,K,fmt);
        CHECK(false); // Native formats have no lossy host quantizer.
    } catch(const std::invalid_argument&) {}
    std::vector<float> decoded(size_t(N)*K);
    std::string err;
    CHECK(gguf_dequantize(type,file.data()+16,size_t(N)*K,decoded.data(),err));
    for(int n=0;n<N;++n)for(int k=0;k<K;++k){
        const float got=w.at(n,k);
        const float ref=decoded[size_t(n)*K+k];
        CHECK(f32_to_bits(got)==f32_to_bits(ref));
        // A BF16 projection staging the native block must equal the
        // Strategy A path when its selected destination is BF16.
        CHECK(f32_to_bf16(got).bits==f32_to_bf16(ref).bits);
        uint32_t golden=u32(file.data()+16+raw_size+(size_t(n)*K+k)*4);
        CHECK(f32_to_bits(got)==golden);
    }
    // The grouped GEMM's B tile is [K/2][256][2] bf16 VNNI. Exercise all
    // scale/min groups and a nonzero N column against the independent decoder.
    for(int n=0;n<N;++n){
        std::vector<uint32_t> tile(size_t(K/2)*256,0xdeadbeefu);
        for(int k=0;k<K;k+=16){
            tile16(fmt,w.payload+size_t(n)*block_bytes,k,
                   tile.data()+size_t(k/2)*256,256,7);
        }
        for(int p=0;p<K/2;++p){
            const uint32_t expected=uint32_t(f32_to_bf16(decoded[size_t(n)*K+2*p]).bits) |
                (uint32_t(f32_to_bf16(decoded[size_t(n)*K+2*p+1]).bits)<<16);
            CHECK(tile[size_t(p)*256+7]==expected);
        }
    }
    // GEMV and GEMM references read the checked-in independent f32 golden,
    // while the runtime operand reads the original compressed device layout.
    constexpr int M=3;
    std::vector<float> x(size_t(M)*K),native(size_t(M)*N),reference(size_t(M)*N);
    for(int m=0;m<M;++m)for(int k=0;k<K;++k)
        x[size_t(m)*K+k]=float(((k*17+m*13)%41)-20)/37.0f;
    for(int m=0;m<M;++m)for(int n=0;n<N;++n){
        for(int k=0;k<K;++k){
            native[size_t(m)*N+n]+=w.at(n,k)*x[size_t(m)*K+k];
            reference[size_t(m)*N+n]+=decoded[size_t(n)*K+k]*x[size_t(m)*K+k];
        }
        CHECK(f32_to_bits(native[size_t(m)*N+n])==f32_to_bits(reference[size_t(m)*N+n]));
    }
    // Two super-blocks in one row must keep their independent local scales.
    std::vector<uint8_t> two(size_t(2)*block_bytes);
    std::memcpy(two.data(),file.data()+16,block_bytes);
    std::memcpy(two.data()+block_bytes,file.data()+16+block_bytes,block_bytes);
    QuantWeight wide{fmt,1,2*K,two.data(),nullptr,nullptr,2*block_bytes,0};
    for(int k=0;k<2*K;++k)
        CHECK(f32_to_bits(wide.at(0,k))==u32(file.data()+16+raw_size+
              size_t(k/K)*K*4+size_t(k%K)*4));
    for(int k=0;k<2*K;k+=16){
        float values[16];
        tile_scaled(fmt,two.data(),k,block_bytes,values);
        for(int i=0;i<16;++i)
            CHECK(f32_to_bf16(values[i]).bits==f32_to_bf16(wide.at(0,k+i)).bits);
    }
}

static void gather_test(int nv){
    constexpr int NK=16,D=128,M=3;
    const int NV=nv,K=NV*D;
    std::vector<float> grouped(size_t(M)*K),tiled(size_t(M)*K),w(K);
    std::vector<bf16_t> grouped_bf(size_t(M)*K),tiled_bf(size_t(M)*K);
    for(size_t i=0;i<grouped.size();++i)grouped[i]=float(int((i*13)%127)-63)/101.0f;
    for(size_t i=0;i<grouped.size();++i)grouped_bf[i]=f32_to_bf16(grouped[i]);
    for(int k=0;k<K;++k)w[k]=float((k*7)%89-44)/71.0f;
    gguf_gather_v_input(grouped.data(),tiled.data(),M,NK,NV,D);
    gguf_gather_v_input(grouped_bf.data(),tiled_bf.data(),M,NK,NV,D);
    for(int m=0;m<M;++m)for(int h=0;h<NV;++h)for(int d=0;d<D;++d){
        const int src=int(gguf_v_grouped_index(h,NK,NV));
        CHECK(tiled[(size_t(m)*NV+h)*D+d]==grouped[(size_t(m)*NV+src)*D+d]);
        CHECK(tiled_bf[(size_t(m)*NV+h)*D+d].bits==
              grouped_bf[(size_t(m)*NV+src)*D+d].bits);
    }
    for(int m=0;m<M;++m){
        float native=0,inverse=0;
        for(int k=0;k<K;++k){
            native+=w[k]*tiled[size_t(m)*K+k];
            const int tiled_col=int(gguf_v_tiled_index(k/D,NK,NV))*D+k%D;
            inverse+=w[tiled_col]*grouped[size_t(m)*K+k];
        }
        CHECK(std::fabs(native-inverse)<0.001f);
    }
}
static void raw_row_test(Fmt fmt,uint32_t type,int row_bytes){
    using namespace gguf_test_writer;
    constexpr int K=256,N=1024,NK=2,NV=4,D=128;
    Bytes raw(size_t(N)*row_bytes);
    for(size_t i=0;i<raw.size();++i)raw[i]=uint8_t((i*11+i/row_bytes)%251);
    const std::string path="/tmp/grimoire-native-row-"+std::to_string(getpid())+".gguf";
    const Bytes file=build_file({{"general.alignment",4,num(64,4)}},
        {{"blk.0.attn_qkv.weight",{K,N},type,0,raw}});
    {std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(file.data()),file.size());}
    Qwen35Model model;model.cfg.lin_k_heads=NK;model.cfg.lin_v_heads=NV;
    model.cfg.lin_k_dim=D;model.cfg.lin_v_dim=D;
    model.gguf_file=std::make_unique<GGUFFile>();
    std::string err;CHECK(model.gguf_file->open(path,err));
    TensorRef r;r.gguf=model.gguf_file->find("blk.0.attn_qkv.weight");
    CHECK(r.gguf!=nullptr);if(!r.gguf){::unlink(path.c_str());return;}
    r.t.name="model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
    r.t.shape={N,K};r.gguf_inverse=TensorRef::GGUFInverse::QKV_ROWS;
    // Logical grouped V head 1 is physical tiled head 2; copy all 144/210
    // block bytes, including scales, without dequantization.
    Bytes copied(row_bytes);
    CHECK(model.read_gguf_raw_rows(r,copied.data(),err,512+D,1));
    const int physical=512+2*D;
    CHECK(std::memcmp(copied.data(),raw.data()+size_t(physical)*row_bytes,row_bytes)==0);
    CHECK(bytes_per_row(fmt,K)==row_bytes);
    // The reader refuses a named raw tensor whose K is not a whole block.
    GGUFTensor bad_tensor=*r.gguf;
    bad_tensor.ne[0]=K-1;
    TensorRef bad_ref=r;
    bad_ref.gguf=&bad_tensor;
    bad_ref.t.shape={N,K-1};
    CHECK(!model.read_gguf_raw_rows(bad_ref,copied.data(),err,0,1));
    CHECK(err.find(bad_tensor.name)!=std::string::npos);
    ::unlink(path.c_str());
}

struct HostQK {
    Fmt fmt;
    int N,K,rb;
    std::vector<uint8_t> raw;
    std::vector<float> independent;
    std::vector<bf16_t> tile;
};
static HostQK host_qk(Fmt fmt,int N,int K){
    const char* name=fmt==Fmt::Q4_K?"Q4_K":fmt==Fmt::Q5_K?"Q5_K":
        fmt==Fmt::Q6_K?"Q6_K":fmt==Fmt::Q8_0?"Q8_0":
        fmt==Fmt::IQ4_XS?"IQ4_XS":"IQ4_NL";
    std::ifstream in(std::string("tests/gguf/data/")+name+".bin",std::ios::binary);
    CHECK(bool(in));
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)),{});
    const int be=native_gguf_block_elems(fmt),rb=bytes_per_row(fmt,be);
    const int src_rows=int(u32(file.data()+8))/be;
    HostQK w{fmt,N,K,rb,{},{},{}};
    w.raw.resize(size_t(N)*rb*(K/be));
    for(int n=0;n<N;++n)for(int b=0;b<K/be;++b)
        std::memcpy(w.raw.data()+(size_t(n)*(K/be)+b)*rb,
                    file.data()+16+size_t((n+b)%src_rows)*rb,rb);
    w.independent.resize(size_t(N)*K);
    std::string err;
    CHECK(gguf_dequantize(native_gguf_type(fmt),
        w.raw.data(),size_t(N)*K,w.independent.data(),err));
    w.tile.resize(size_t(N)*K);
    for(int n=0;n<N;++n)for(int b=0;b<K;b+=16){
        uint32_t packed[8];
        tile16(fmt,w.raw.data()+size_t(n)*rb*(K/be),b,packed,1,0);
        for(int p=0;p<8;++p){
            w.tile[size_t(n)*K+b+2*p]={uint16_t(packed[p])};
            w.tile[size_t(n)*K+b+2*p+1]={uint16_t(packed[p]>>16)};
        }
    }
    return w;
}
template <Fmt F, int MB>
static void fast_gemv_case(int K,int mc){
    constexpr int N=3;
    const HostQK w=host_qk(F,N,K);
    std::vector<float> x(size_t(MB)*K);
    for(int m=0;m<mc;++m)for(int k=0;k<K;++k)
        x[size_t(m)*K+k]=float(((k*37+m*53+k/NativeGGUFBlock<F>::elems*11)%211)-105)/173.0f;
    const float* xp[MB];
    for(int m=0;m<MB;++m)xp[m]=x.data()+size_t(m)*K;
    for(int n=0;n<N;++n){
        const uint8_t* row=w.raw.data()+size_t(n)*w.rb*(K/NativeGGUFBlock<F>::elems);
        float got[MB]={};
        for(int lane=0;lane<16;++lane){
            float part[MB]={};
            native_qk_row_lane<F,MB>(row,xp,K,lane,mc,part);
            for(int m=0;m<mc;++m)got[m]+=part[m];
        }
        for(int m=0;m<mc;++m){
            float ref=0.0f;
            for(int k=0;k<K;++k)
                ref=std::fma(w.independent[size_t(n)*K+k],x[size_t(m)*K+k],ref);
            CHECK(std::fabs(got[m]-ref)<=2e-3f*std::max(1.0f,std::fabs(ref)));
        }
    }
}
static void fast_gemv_test(){
    for(int K:{512,2048,17408}){
        fast_gemv_case<Fmt::Q4_K,1>(K,1);
        fast_gemv_case<Fmt::Q6_K,1>(K,1);
        fast_gemv_case<Fmt::Q5_K,1>(K,1);
        fast_gemv_case<Fmt::Q8_0,1>(K,1);
        fast_gemv_case<Fmt::IQ4_XS,1>(K,1);
        fast_gemv_case<Fmt::IQ4_NL,1>(K,1);
    }
    fast_gemv_case<Fmt::Q4_K,4>(2048,4);
    fast_gemv_case<Fmt::Q6_K,4>(2048,4);
    fast_gemv_case<Fmt::Q5_K,4>(2048,4);
    fast_gemv_case<Fmt::Q8_0,4>(2048,4);
    fast_gemv_case<Fmt::IQ4_XS,4>(2048,4);
    fast_gemv_case<Fmt::IQ4_NL,4>(2048,4);
    // Isolate the terms a decoder can accidentally omit: Q4_K's dmin
    // subtraction and Q6_K's signed -32 offset.
    for(Fmt fmt:{Fmt::Q4_K,Fmt::Q6_K}){
        std::vector<uint8_t> block(fmt==Fmt::Q4_K?144:210,0);
        const float expected=fmt==Fmt::Q4_K?-16.0f:-512.0f;
        if(fmt==Fmt::Q4_K){
            block[0]=0x00;block[1]=0x3c; // d=1
            block[2]=0x00;block[3]=0x38; // dmin=0.5
            block[4]=1;block[8]=1;       // first group scale=1, min=1
        }else{
            block[192]=1;block[208]=0x00;block[209]=0x3c; // sc0=1, d=1
        }
        float x[256],out[256];std::fill(x,x+256,1.0f);
        std::string err;
        CHECK(gguf_dequantize(fmt==Fmt::Q4_K?12:14,block.data(),256,out,err));
        float ref=0.0f;for(int k=0;k<256;++k)ref+=out[k]*x[k];
        float fast=0.0f;const float* xp[1]={x};
        for(int lane=0;lane<16;++lane){
            float part=0.0f;
            if(fmt==Fmt::Q4_K)
                native_qk_row_lane<Fmt::Q4_K,1>(block.data(),xp,256,lane,1,&part);
            else
                native_qk_row_lane<Fmt::Q6_K,1>(block.data(),xp,256,lane,1,&part);
            fast+=part;
        }
        CHECK(ref==expected);
        CHECK(fast==expected);
    }
}
static float host_w(const HostQK& w,int n,int k,bool grouped){
    const size_t i=size_t(n)*w.K+k;
    return bf16_to_f32(grouped?w.tile[i]:f32_to_bf16(w.independent[i]));
}
static std::vector<float> host_moe(const HostQK& gu,const HostQK& dn,bool grouped){
    constexpr int E=3,TOP=2,H=256,I=256,M=4,R=M*TOP;
    const int routes[R]={2,0,1,2,0,1,2,1};
    const float weights[R]={0.7f,0.3f,0.4f,0.6f,0.2f,0.8f,0.5f,0.5f};
    std::vector<bf16_t> x(size_t(M)*H);
    for(int m=0;m<M;++m)for(int k=0;k<H;++k)
        x[size_t(m)*H+k]=f32_to_bf16(float((m*17+k*3)%37-18)/1024.0f);
    int order[R],inverse[R];
    if(grouped){
        int count[E]={},off[E+1]={},next[E];
        for(int r=0;r<R;++r)++count[routes[r]];
        for(int e=0;e<E;++e)off[e+1]=off[e]+count[e];
        std::copy(off,off+E,next);
        for(int r=0;r<R;++r){const int p=next[routes[r]]++;order[p]=r;inverse[r]=p;}
    }else for(int r=0;r<R;++r){order[r]=r;inverse[r]=r;}
    std::vector<float> rows(size_t(R)*H),out(size_t(M)*H,0.0f);
    for(int p=0;p<R;++p){
        const int r=order[p],e=routes[r],m=r/TOP;
        std::vector<bf16_t> hidden(I);
        for(int j=0;j<I;++j){
            float gate=0,up=0;
            for(int k=0;k<H;++k){
                const float a=bf16_to_f32(x[size_t(m)*H+k]);
                gate+=a*host_w(gu,e*2*I+j,k,grouped);
                up+=a*host_w(gu,e*2*I+I+j,k,grouped);
            }
            hidden[j]=f32_to_bf16((gate/(1.0f+std::exp(-gate)))*up);
        }
        for(int j=0;j<H;++j){
            float v=0;
            for(int k=0;k<I;++k)
                v+=bf16_to_f32(hidden[k])*host_w(dn,e*H+j,k,grouped);
            rows[size_t(p)*H+j]=v;
        }
    }
    for(int r=0;r<R;++r)for(int j=0;j<H;++j)
        out[size_t(r/TOP)*H+j]+=weights[r]*rows[size_t(inverse[r])*H+j];
    return out;
}
static void grouped_moe_test(){
    constexpr int E=3,H=256,I=256;
    for(const Fmt fmt:{Fmt::Q4_K,Fmt::Q5_K,Fmt::Q6_K,Fmt::Q8_0,Fmt::IQ4_XS,Fmt::IQ4_NL}){
        const HostQK gu=host_qk(fmt,E*2*I,H);
        const HostQK dn=host_qk(fmt,E*H,I);
        const auto grouped=host_moe(gu,dn,true),per_expert=host_moe(gu,dn,false);
        for(size_t i=0;i<grouped.size();++i)
            CHECK(std::fabs(grouped[i]-per_expert[i])<=
                  1e-5f*std::max(1.0f,std::fabs(per_expert[i])));
    }
}
static void dense_ffn_raw_test(Fmt fmt,uint32_t type){
    using namespace gguf_test_writer;
    constexpr int H=5120,I=17408,N=2;
    const int gu_row=bytes_per_row(fmt,H),dn_row=bytes_per_row(fmt,I);
    CHECK(gu_row>0&&dn_row>0);
    Bytes gate(size_t(N)*gu_row),up(size_t(N)*gu_row),down(size_t(N)*dn_row);
    for(size_t i=0;i<gate.size();++i){gate[i]=uint8_t((i*7+1)%251);up[i]=uint8_t((i*11+2)%251);}
    for(size_t i=0;i<down.size();++i)down[i]=uint8_t((i*13+3)%251);
    const uint64_t up_off=(gate.size()+63)&~uint64_t(63);
    const uint64_t dn_off=(up_off+up.size()+63)&~uint64_t(63);
    const std::string path="/tmp/grimoire-dense-ffn-"+std::to_string(getpid())+
                           "-"+std::to_string(type)+".gguf";
    const Bytes file=build_file({{"general.alignment",4,num(64,4)}},
        {{"blk.0.ffn_gate.weight",{H,N},type,0,gate},
         {"blk.0.ffn_up.weight",{H,N},type,up_off,up},
         {"blk.0.ffn_down.weight",{I,N},type,dn_off,down}});
    {std::ofstream out(path,std::ios::binary);
     out.write(reinterpret_cast<const char*>(file.data()),file.size());}
    Qwen35Model m;m.cfg.n_layers=1;m.cfg.lin_k_heads=16;m.cfg.lin_v_heads=48;
    m.cfg.lin_k_dim=128;m.cfg.lin_v_dim=128;
    m.gguf_file=std::make_unique<GGUFFile>();
    std::string err;CHECK(m.gguf_file->open(path,err));
    const std::string base="model.language_model.layers.0.mlp.";
    auto ref=[&](const char* role,const char* proj,int K){
        TensorRef r;r.gguf=m.gguf_file->find(std::string("blk.0.ffn_")+role+".weight");
        r.t.name=base+proj+"_proj.weight";r.t.shape={N,K};return r;};
    const TensorRef g=ref("gate","gate",H),u=ref("up","up",H),d=ref("down","down",I);
    CHECK(g.gguf&&u.gguf&&d.gguf);
    CHECK(!keep_qwen_bf16(g.t.name)&&!keep_qwen_bf16(u.t.name)&&
          !keep_qwen_bf16(d.t.name));
    CHECK(g.t.shape==std::vector<int64_t>({N,H})&&
          u.t.shape==std::vector<int64_t>({N,H})&&
          d.t.shape==std::vector<int64_t>({N,I}));
    // concat_upload_t's native branch uses these reads in gate-then-up row order.
    Bytes fused(gate.size()+up.size()),got_down(down.size());
    CHECK(m.read_gguf_raw_rows(g,fused.data(),err));
    CHECK(m.read_gguf_raw_rows(u,fused.data()+gate.size(),err));
    CHECK(m.read_gguf_raw_rows(d,got_down.data(),err));
    CHECK(std::memcmp(fused.data(),gate.data(),gate.size())==0);
    CHECK(std::memcmp(fused.data()+gate.size(),up.data(),up.size())==0);
    CHECK(got_down==down);
    ::unlink(path.c_str());
}
int main(){
    format_test(Fmt::Q4_K,12,"Q4_K",144);
    format_test(Fmt::Q6_K,14,"Q6_K",210);
    format_test(Fmt::Q5_K,13,"Q5_K",176);
    format_test(Fmt::Q8_0,8,"Q8_0",34);
    format_test(Fmt::IQ4_XS,23,"IQ4_XS",136);
    format_test(Fmt::IQ4_NL,20,"IQ4_NL",18);
    gather_test(32);
    gather_test(48);
    raw_row_test(Fmt::Q4_K,12,144);
    raw_row_test(Fmt::Q6_K,14,210);
    raw_row_test(Fmt::Q5_K,13,176);
    raw_row_test(Fmt::Q8_0,8,272);
    raw_row_test(Fmt::IQ4_XS,23,136);
    raw_row_test(Fmt::IQ4_NL,20,144);
    grouped_moe_test();
    fast_gemv_test();
    dense_ffn_raw_test(Fmt::Q4_K,12);
    dense_ffn_raw_test(Fmt::Q6_K,14);
    dense_ffn_raw_test(Fmt::Q5_K,13);
    dense_ffn_raw_test(Fmt::Q8_0,8);
    dense_ffn_raw_test(Fmt::IQ4_XS,23);
    dense_ffn_raw_test(Fmt::IQ4_NL,20);
    std::printf("GGUF native: %d assertions, %d failures\n",checked,bad);
    return bad?1:0;
}
