// Host verification of the exact decoder used in SYCL kernels.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/weights.hpp"
#include "b70/gguf_decode.hpp"
#include "b70/gguf_qwen35.hpp"
#include "b70/native_model.hpp"
#include "gguf/test_writer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include <unistd.h>

using namespace b70;
static int bad=0,checked=0;
#define CHECK(c) do { ++checked; if(!(c)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#c);++bad;} } while(0)
static uint32_t u32(const uint8_t* p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}

static void format_test(Fmt fmt,uint32_t type,const char* name,int block_bytes){
    std::ifstream in(std::string("tests/gguf/data/")+name+".bin",std::ios::binary);
    CHECK(bool(in));if(!in)return;
    std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)),{});
    CHECK(file.size()>=16);if(file.size()<16)return;
    const int K=256,N=int(u32(file.data()+8))/K;
    const uint32_t raw_size=u32(file.data()+12);
    CHECK(u32(file.data()+4)==type);
    CHECK(raw_size==uint32_t(N*block_bytes));
    CHECK(file.size()==16+raw_size+size_t(N)*K*4);
    CHECK(bytes_per_row(fmt,K)==block_bytes);
    CHECK(bytes_per_row(fmt,255)==0);
    CHECK(scales_per_row(fmt,K)==0);
    CHECK(bits_per_elem(fmt)==double(block_bytes)*8/K);
    CHECK(std::strcmp(fmt_name(fmt),name==std::string("Q4_K")?"q4_k":"q6_k")==0);
    QuantWeight w{fmt,N,K,file.data()+16,nullptr,nullptr,block_bytes,0};
    CHECK(w.scales==nullptr && w.zeros==nullptr && w.row_scales==0);
    const QuantWeight sliced=slice_quant_rows(w,1,1);
    CHECK(sliced.N==1 && sliced.K==K && sliced.fmt==fmt);
    CHECK(sliced.payload==w.payload+block_bytes && sliced.scales==nullptr);
    for(int k=0;k<K;++k)CHECK(f32_to_bits(sliced.at(0,k))==
                            u32(file.data()+16+raw_size+(size_t(K)+k)*4));
    try {
        float dummy[256]{};
        (void)quantize(dummy,1,256,fmt);
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
    QuantWeight wide{fmt,1,512,two.data(),nullptr,nullptr,2*block_bytes,0};
    for(int k=0;k<512;++k)
        CHECK(f32_to_bits(wide.at(0,k))==u32(file.data()+16+raw_size+
              size_t(k/256)*256*4+size_t(k%256)*4));
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
    ::unlink(path.c_str());
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
    gather_test(32);
    gather_test(48);
    raw_row_test(Fmt::Q4_K,12,144);
    raw_row_test(Fmt::Q6_K,14,210);
    dense_ffn_raw_test(Fmt::Q4_K,12);
    dense_ffn_raw_test(Fmt::Q6_K,14);
    std::printf("GGUF native: %d assertions, %d failures\n",checked,bad);
    return bad?1:0;
}
