#include "b70/qwen35.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace b70;
namespace fs=std::filesystem;
static void check(bool b,const std::string& why){if(!b)throw std::runtime_error(why);}
struct Temp {fs::path path;Temp(){char p[]="/tmp/grimoire-native-XXXXXX";check(::mkdtemp(p),"mkdtemp");path=p;}
    ~Temp(){std::error_code e;fs::remove_all(path,e);}};
struct Tensor {std::string name;std::vector<int64_t> shape;std::vector<bf16_t> values;};
std::vector<Tensor> fixture(){
    std::vector<Tensor> ts;
    auto add=[&](std::string name,std::vector<int64_t> shape){
        Tensor t{"model.language_model."+name,shape,{}};size_t n=1;for(auto d:shape)n*=size_t(d);t.values.resize(n);
        // Distinct expert/row/group values. Never compare just identical zero tensors.
        for(size_t i=0;i<n;++i)t.values[i]=f32_to_bf16(float(int((i/128)%7)-3)*0.25f+float(i%127)/128);
        ts.push_back(std::move(t));
    };
    add("embed_tokens.weight",{256,128});add("norm.weight",{128});add("lm_head.weight",{256,128});
    add("layers.0.input_layernorm.weight",{128});add("layers.0.post_attention_layernorm.weight",{128});
    for(auto n:{"q_proj","k_proj","v_proj","o_proj"})add(std::string("layers.0.self_attn.")+n+".weight",{128,128});
    add("layers.0.self_attn.q_norm.weight",{64});add("layers.0.self_attn.k_norm.weight",{64});
    add("layers.0.mlp.gate.weight",{2,128});add("layers.0.mlp.shared_expert_gate.weight",{1,128});
    for(auto n:{"gate_proj","up_proj","down_proj"})add(std::string("layers.0.mlp.shared_expert.")+n+".weight",{128,128});
    add("layers.0.mlp.experts.gate_up_proj",{2,256,128});add("layers.0.mlp.experts.down_proj",{2,128,128});
    add("layers.0.linear_attn.in_proj_a.weight",{2,128});add("layers.0.linear_attn.in_proj_b.weight",{2,128});
    add("unaligned.weight",{2,129});
    return ts;
}
void write_source(const fs::path& dir,const std::vector<Tensor>& ts){
    fs::create_directories(dir);
    std::ofstream(dir/"config.json")<<R"({"model_type":"qwen3_5_moe","text_config":{"hidden_size":128,"num_hidden_layers":1,"vocab_size":256,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":64,"num_experts":2,"num_experts_per_tok":1,"moe_intermediate_size":128,"shared_expert_intermediate_size":128,"layer_types":["full_attention"]}})";
    std::ofstream(dir/"tokenizer.json")<<"{}\n"; // copied, not used for inference in this fixture
    std::ofstream(dir/"tokenizer_config.json")<<"{\"chat_template\":\"fixture\"}\n";
    std::ostringstream header;header<<'{';uint64_t off=0;
    for(size_t i=0;i<ts.size();++i){const auto& t=ts[i];if(i)header<<',';
        header<<'"'<<t.name<<"\":{\"dtype\":\"BF16\",\"shape\":[";
        for(size_t d=0;d<t.shape.size();++d){if(d)header<<',';header<<t.shape[d];}
        header<<"],\"data_offsets\":["<<off<<',';off+=t.values.size()*2;header<<off<<"]}";}
    header<<'}';std::string h=header.str();while(h.size()%8)h+=' ';
    std::ofstream f(dir/"model.safetensors",std::ios::binary);const uint64_t n=h.size();f.write((const char*)&n,8);f<<h;
    for(const auto& t:ts)f.write((const char*)t.values.data(),std::streamsize(t.values.size()*2));
    f.close();check(bool(f),"write safetensors fixture");
}
std::vector<char> read_file(const fs::path& p){std::ifstream f(p,std::ios::binary);return {std::istreambuf_iterator<char>(f),{}};}
int run(const fs::path& exe,std::vector<std::string> args,const fs::path& log){
    args.insert(args.begin(),exe.string());std::vector<char*> a;for(auto& s:args)a.push_back(s.data());a.push_back(nullptr);
    const pid_t pid=::fork();check(pid>=0,"fork");
    if(pid==0){int fd=::open(log.c_str(),O_WRONLY|O_CREAT|O_TRUNC,0600);if(fd<0)_exit(126);
        ::dup2(fd,1);::dup2(fd,2);::close(fd);::execv(exe.c_str(),a.data());_exit(127);}
    int s=0;check(::waitpid(pid,&s,0)==pid,"waitpid");return WIFEXITED(s)?WEXITSTATUS(s):128;
}
void malformed(const fs::path& good,const fs::path& bad,const std::string& label,
               const std::function<void(std::vector<char>&)>& mutate){
    auto b=read_file(good);mutate(b);std::ofstream f(bad,std::ios::binary);f.write(b.data(),b.size());f.close();
    NativeModel m;std::string e;check(!m.open(bad.string(),e),"accepted malformed file: "+label);check(!e.empty(),"missing diagnostic: "+label);
}
int main(int argc,char** argv){
    try{
        Temp temp;const auto exe=fs::absolute(argc>1?argv[1]:"bin/grimoire-quantize");
        const auto source=temp.path/"source-bf16",log=temp.path/"converter.log";
        const auto ts=fixture();write_source(source,ts);
        const auto original=read_file(source/"model.safetensors");
        struct Format{const char* name;Fmt fmt;};
        const Format formats[]={{"int4",Fmt::INT4},{"fp8",Fmt::FP8_E4M3},{"fp8-e5m2",Fmt::FP8_E5M2},
            {"int8",Fmt::INT8},{"mxfp4",Fmt::MXFP4},{"mxfp8",Fmt::MXFP8},{"bf16",Fmt::BF16}};
        for(const auto& format:formats){
            const auto out=temp.path/format.name;
            if(run(exe,{source.string(),out.string(),"--format",format.name,"--policy","linear","--bundle"},log)!=0){
                const auto msg=read_file(log);throw std::runtime_error(std::string(msg.begin(),msg.end()));}
            check(read_file(out/"config.json")==read_file(source/"config.json"),"config changed");
            check(read_file(out/"tokenizer_config.json")==read_file(source/"tokenizer_config.json"),"chat template changed");
            Qwen35Model model;std::string e;
            check(model.load(out.string(),e),"reload: "+e);
            check(model.native_model && model.shards.empty(),"bundle depends on BF16 shards");
            for(const auto& t:ts){
                const auto* r=model.native_model->find(t.name);check(r,"missing exported tensor");
                if(r->encoding==uint32_t(NativeEncoding::RAW)){
                    check(r->payload_bytes==t.values.size()*2,"RAW length changed");
                    check(std::memcmp(model.native_model->payload(*r),t.values.data(),r->payload_bytes)==0,"RAW bytes changed");
                }else{
                    std::vector<float> fp(t.values.size());for(size_t i=0;i<fp.size();++i)fp[i]=bf16_to_f32(t.values[i]);
                    const int K=int(t.shape.back()),N=int(fp.size()/K);const auto q=quantize(fp.data(),N,K,format.fmt);
                    QuantWeight v;check(model.native_model->view(*r,0,N,v,e),"packed view: "+e);const auto copy=copy_packed(v);
                    check(copy.fmt==format.fmt && copy.payload==q.payload && copy.scales_raw==q.scales_raw && copy.zeros==q.zeros,
                          "saved format or payload changed on reload");
                    if (N>1) {
                        const auto tail=slice_quant_rows(v,N-1,1);
                        for(int k=0;k<K;++k)check(tail.at(0,k)==v.at(N-1,k),"prefill/TP scale slice stride");
                    }
                }
            }
            for(const auto& name:{"layers.0.mlp.gate.weight","layers.0.mlp.shared_expert_gate.weight",
                    "layers.0.linear_attn.in_proj_a.weight","layers.0.linear_attn.in_proj_b.weight","lm_head.weight"}){
                const auto* r=model.native_model->find(std::string("model.language_model.")+name);
                check(r && r->encoding==0,"sensitive tensor quantized");}
            // Exercise the SAME native_view/read_native_f32 methods used during GPU upload.
            const auto& layer=model.layers.at(0);
            const Tensor* gate=nullptr;const Tensor* down=nullptr;
            for(const auto& t:ts){if(t.name.find("experts.gate_up_proj")!=std::string::npos)gate=&t;
                if(t.name.find("experts.down_proj")!=std::string::npos)down=&t;}
            for(int expert=0;expert<2;++expert)for(int part=0;part<3;++part){
                const TensorRef& ref=part==0?layer.e_gate_p[expert]:part==1?layer.e_up_p[expert]:layer.e_down_p[expert];
                QuantWeight v;check(model.native_view(ref,v,e),"expert slice: "+e);check(v.fmt==format.fmt,"expert format changed");
                const Tensor* src=part==2?down:gate;const size_t first=part==2?size_t(expert)*128:size_t(expert)*256+size_t(part)*128;
                std::vector<float> fp(src->values.size());for(size_t i=0;i<fp.size();++i)fp[i]=bf16_to_f32(src->values[i]);
                auto expected=quantize(fp.data()+first*128,128,128,format.fmt);
                const auto actual=copy_packed(v);check(actual.payload==expected.payload && actual.scales_raw==expected.scales_raw && actual.zeros==expected.zeros,
                    "wrong fused expert slice or zero-plane offset");
                std::vector<float> decoded(128*128);check(model.read_native_f32(ref,decoded.data(),e),"expert float decode");
                for(int n=0;n<128;++n)for(int k=0;k<128;++k)check(decoded[n*128+k]==expected.view().at(n,k),"expert value changed");
                check(model.bytes(ref)==int64_t(expected.payload.size()),"slice byte count is whole tensor");
            }
            const auto before=read_file(out/"model-v3.b70");
            check(run(exe,{source.string(),out.string(),"--bundle"},log)!=0,"existing bundle overwritten");
            check(before==read_file(out/"model-v3.b70"),"existing bundle modified");
            check(run(exe,{source.string(),(out/"model-v3.b70").string()},log)!=0,"existing file overwritten");
            check(before==read_file(out/"model-v3.b70"),"existing file modified");
            std::cout<<"PASS "<<format.name<<": convert, independent reload, byte preservation, all expert slices, overwrite protection\n";
        }
        const auto defaults=temp.path/"defaults";
        check(run(exe,{source.string(),defaults.string(),"--bundle"},log)==0,"default export");
        NativeModel dm;std::string de;check(dm.open((defaults/"model-v3.b70").string(),de),"default reload");
        check(dm.find("model.language_model.layers.0.self_attn.q_proj.weight")->encoding==0,"expert policy quantized attention");
        check(dm.find("model.language_model.layers.0.mlp.experts.gate_up_proj")->encoding==2,"default expert format is not INT4");
        // Exceed the converter's 32 MiB FP32 window by three rows. Verify
        // both sides of the chunk boundary, including INT4's separate zeros.
        auto big=ts;Tensor extra{"model.language_model.layers.0.mlp.experts.extra.weight",{65539,128},{}};
        extra.values.resize(size_t(65539)*128);
        for(size_t i=0;i<extra.values.size();++i)extra.values[i]=f32_to_bf16(float(int((i/128)%9)-4)+float(i%128)/128);
        big.push_back(std::move(extra));write_source(temp.path/"large",big);
        for(const auto& f:std::vector<Format>{{"int4",Fmt::INT4},{"fp8",Fmt::FP8_E4M3}}){
            const auto out=temp.path/(std::string("large-")+f.name+".b70");
            check(run(exe,{(temp.path/"large").string(),out.string(),"--format",f.name},log)==0,"chunked export");
            NativeModel m;std::string e;check(m.open(out.string(),e),"chunked reload: "+e);
            const auto* r=m.find(big.back().name);check(r,"chunked tensor missing");
            for(int row:{0,65535,65536,65538}){
                QuantWeight w;check(m.view(*r,row,1,w,e),"chunk row slice");float fp[128];
                for(int k=0;k<128;++k)fp[k]=bf16_to_f32(big.back().values[size_t(row)*128+k]);
                const auto expected=quantize(fp,1,128,f.fmt),actual=copy_packed(w);
                check(actual.payload==expected.payload && actual.scales_raw==expected.scales_raw && actual.zeros==expected.zeros,
                    "chunk boundary changed weight/scale/zero bytes");
            }
        }
        std::cout<<"PASS expert-only default and streaming chunk boundaries (INT4/FP8)\n";
        const auto good=temp.path/"int4/model-v3.b70",bad=temp.path/"bad.b70";
        auto record=[](std::vector<char>& b)->NativeTensorRecord*{NativeFileHeader h;std::memcpy(&h,b.data(),sizeof h);
            return reinterpret_cast<NativeTensorRecord*>(b.data()+h.toc_offset);};
        malformed(good,bad,"truncated",[](auto& b){b.resize(12);});
        malformed(good,bad,"TOC overflow",[](auto& b){auto* h=reinterpret_cast<NativeFileHeader*>(b.data());h->toc_offset=UINT64_MAX;});
        malformed(good,bad,"offset overflow",[&](auto& b){record(b)[0].payload_offset=UINT64_MAX-4095;});
        malformed(good,bad,"rank",[&](auto& b){record(b)[0].rank=5;});
        malformed(good,bad,"shape overflow",[&](auto& b){auto& r=record(b)[0];r.rank=2;r.shape[0]=INT64_MAX;r.shape[1]=INT64_MAX;});
        malformed(good,bad,"dtype",[&](auto& b){record(b)[0].source_dtype=999;});
        malformed(good,bad,"duplicate name",[&](auto& b){std::memcpy(record(b)[1].name,record(b)[0].name,192);});
        malformed(good,bad,"unterminated name",[&](auto& b){std::memset(record(b)[0].name,'x',192);});
        auto packed=[&](auto& b)->NativeTensorRecord&{auto* r=record(b);for(size_t i=0;;++i)if(r[i].encoding==2)return r[i];};
        malformed(good,bad,"unknown encoding",[&](auto& b){packed(b).encoding=999;});
        malformed(good,bad,"wrong group bytes",[&](auto& b){packed(b).scales_bytes--;});
        malformed(good,bad,"zero scale",[&](auto& b){auto& r=packed(b);b[r.scales_offset]=b[r.scales_offset+1]=0;});
        malformed(good,bad,"invalid zero",[&](auto& b){auto& r=packed(b);b[r.scales_offset+r.scales_bytes-1]=char(16);});
        malformed(good,bad,"overlap",[&](auto& b){auto& r=packed(b);r.payload_offset=record(b)[0].payload_offset;});
        // v2 files used the same ABI but only RAW/MXFP4 encodings.
        auto legacy=read_file(temp.path/"mxfp4/model-v3.b70");reinterpret_cast<NativeFileHeader*>(legacy.data())->version=2;
        std::ofstream lf(bad,std::ios::binary);lf.write(legacy.data(),legacy.size());lf.close();
        NativeModel lm;std::string e;check(lm.open(bad.string(),e),"v2 compatibility: "+e);
        auto broken=ts;broken.back().values[0]=bf16_t{0x7fc0};write_source(temp.path/"nan",broken);
        const auto failed=temp.path/"failed";
        check(run(exe,{(temp.path/"nan").string(),failed.string(),"--format","bf16","--bundle"},log)!=0,"NaN source accepted");
        check(!fs::exists(failed),"failed conversion published output");
        for(const auto& p:fs::directory_iterator(temp.path))check(p.path().filename().string().find(".tmp-")==std::string::npos,"temporary export leaked");
        check(read_file(source/"model.safetensors")==original,"source checkpoint modified");
        std::cout<<"PASS malformed inputs, v2 compatibility, failed-export cleanup, source preservation\nALL PASS\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
