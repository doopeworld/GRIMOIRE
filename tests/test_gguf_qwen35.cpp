// GRIMOIRE -- host-only qwen35moe GGUF adapter tests.
// Independent forward permutation from conversion/qwen.py:454-646, documented
// as T4 in docs/gguf/FORMAT_SUPPORT.md. T1/T2 forward use lines 396-403.
#include "b70/gguf_qwen35.hpp"
#include "b70/json.hpp"
#include "b70/formats.hpp"
#include "gguf/test_writer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sys/stat.h>
#include <unistd.h>
using namespace b70;
using namespace gguf_test_writer;
static int checks=0,failures=0;
#define CHECK(x) do{++checks;if(!(x)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);++failures;}}while(0)
static std::string tmp(std::string s){return "/tmp/gguf-qwen35-"+std::to_string(getpid())+"-"+s;}
static void write(std::string p,const Bytes& b){std::ofstream f(p,std::ios::binary);f.write((const char*)b.data(),b.size());}
static Bytes raw(const std::vector<float>& v){Bytes b(v.size()*4);std::memcpy(b.data(),v.data(),b.size());return b;}
static KV u32(std::string k,uint32_t v){return {k,4,num(v,4)};}
static KV f32kv(std::string k,float v){uint32_t n;std::memcpy(&n,&v,4);return {k,6,num(n,4)};}
static KV u32arr(std::string k,std::vector<uint32_t> v){Bytes b;put(b,4,4);put(b,v.size(),8);for(auto x:v)put(b,x,4);return {k,9,b};}
static std::vector<KV> metadata(int nk,int nv,int h,int hd,int layers,int experts,int inter){
    const std::string p="qwen35moe.";
    return {u32("general.alignment",64),kv_string("general.architecture","qwen35moe"),
      u32(p+"block_count",layers+1),u32(p+"nextn_predict_layers",1),
      u32(p+"embedding_length",h),u32(p+"attention.head_count",nk),
      u32(p+"attention.head_count_kv",1),u32(p+"attention.key_length",hd),
      u32(p+"attention.value_length",hd),u32(p+"rope.dimension_count",hd),
      u32arr(p+"rope.dimension_sections",{1,1,0,0}),f32kv(p+"rope.freq_base",1e7f),
      f32kv(p+"attention.layer_norm_rms_epsilon",1e-6f),
      u32(p+"expert_count",experts),u32(p+"expert_used_count",1),
      u32(p+"expert_feed_forward_length",inter),u32(p+"expert_shared_feed_forward_length",inter),
      u32(p+"ssm.group_count",nk),u32(p+"ssm.time_step_rank",nv),
      u32(p+"ssm.state_size",2),u32(p+"ssm.inner_size",nv*2),
      u32(p+"ssm.conv_kernel",4),u32(p+"full_attention_interval",2),
      kv_array_strings("tokenizer.ggml.tokens",{"a","b","c","d"}),
      u32("tokenizer.ggml.eos_token_id",3),u32("tokenizer.ggml.bos_token_id",2)};
}
static void add(std::vector<TS>& ts,std::string name,std::vector<uint64_t> dims,std::vector<float> v){
    uint64_t off=ts.empty()?0:ts.back().offset+ts.back().raw.size();off=(off+63)&~uint64_t(63);
    ts.push_back({std::move(name),std::move(dims),0,off,raw(v)});
}
static std::vector<float> seq(size_t n,float base=0){std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=base+float(i)/8;return v;}
static bool same(const std::vector<float>& a,const std::vector<float>& b){return a.size()==b.size()&&!std::memcmp(a.data(),b.data(),a.size()*4);}
// Forward: gguf[tiled j*NK+g] = hf[grouped g*r+j].
static std::vector<float> fwd_rows(const std::vector<float>& hf,int K,int nk,int nv,int width,int prefix){
    std::vector<float> out(hf.size());
    for(int i=0;i<prefix;++i)std::memcpy(out.data()+size_t(i)*K,hf.data()+size_t(i)*K,size_t(K)*4);
    for(int t=0;t<nv;++t){int g=(t%nk)*(nv/nk)+t/nk;
      for(int d=0;d<width;++d)std::memcpy(out.data()+size_t(prefix+t*width+d)*K,
          hf.data()+size_t(prefix+g*width+d)*K,size_t(K)*4);}
    return out;
}
static std::vector<float> fwd_cols(const std::vector<float>& hf,int N,int K,int nk,int nv,int width){
    std::vector<float> out(hf.size());
    for(int n=0;n<N;++n)for(int t=0;t<nv;++t){int g=(t%nk)*(nv/nk)+t/nk;
      for(int d=0;d<width;++d)out[size_t(n)*K+t*width+d]=hf[size_t(n)*K+g*width+d];}
    return out;
}
static void transforms(int nk,int nv){
    int h=nk*4,dk=2,dv=2,qk=2*nk*dk,c=qk+nv*dv;
    std::mt19937 rng{unsigned(nk)};std::uniform_int_distribution<int> dis(-1000,1000);
    auto rand=[&](size_t n){std::vector<float> v(n);for(auto& x:v)x=float(dis(rng))/128;return v;};
    std::vector<TS> ts;
    struct Case{std::string name;std::vector<float> hf;};std::vector<Case> cases;
    auto rows=[&](std::string gg,std::string hf,int K,int N,int w,int pre){auto v=rand(size_t(K)*N);
      add(ts,"blk.0."+gg,{uint64_t(K),uint64_t(N)},fwd_rows(v,K,nk,nv,w,pre));
      cases.push_back({"model.language_model.layers.0."+hf,v});};
    rows("attn_qkv.weight","linear_attn.in_proj_qkv.weight",h,c,dv,qk);
    rows("attn_gate.weight","linear_attn.in_proj_z.weight",h,nv*dv,dv,0);
    rows("ssm_alpha.weight","linear_attn.in_proj_a.weight",h,nv,1,0);
    rows("ssm_beta.weight","linear_attn.in_proj_b.weight",h,nv,1,0);
    rows("ssm_conv1d.weight","linear_attn.conv1d.weight",4,c,dv,qk);
    {auto v=rand(size_t(h)*nv*dv);add(ts,"blk.0.ssm_out.weight",{uint64_t(nv*dv),uint64_t(h)},fwd_cols(v,h,nv*dv,nk,nv,dv));
     cases.push_back({"model.language_model.layers.0.linear_attn.out_proj.weight",v});}
    {auto v=rand(nv);add(ts,"blk.0.ssm_dt.bias",{uint64_t(nv)},fwd_cols(v,1,nv,nk,nv,1));
     cases.push_back({"model.language_model.layers.0.linear_attn.dt_bias",v});}
    {std::vector<float> w(nv),a(nv);for(int i=0;i<nv;++i){w[i]=bf16_to_f32(f32_to_bf16(float(i-4)/16));a[i]=float(-std::exp(double(w[i])));}
     add(ts,"blk.0.ssm_a",{uint64_t(nv)},fwd_cols(a,1,nv,nk,nv,1));
     cases.push_back({"model.language_model.layers.0.linear_attn.A_log",w});}
    {std::vector<float> w(h),g(h);for(int i=0;i<h;++i){w[i]=bf16_to_f32(f32_to_bf16(float(i)/64));g[i]=1+w[i];}
     add(ts,"blk.0.attn_norm.weight",{uint64_t(h)},g);
     cases.push_back({"model.language_model.layers.0.input_layernorm.weight",w});}
    std::string file=tmp("permutation"+std::to_string(nk)+".gguf");
    write(file,build_file({u32("general.alignment",64)},ts));
    Qwen35Model m;m.cfg.n_layers=1;m.cfg.lin_k_heads=nk;m.cfg.lin_v_heads=nv;
    m.cfg.lin_k_dim=dk;m.cfg.lin_v_dim=dv;m.gguf_file=std::make_unique<GGUFFile>();
    std::string err;CHECK(m.gguf_file->open(file,err));CHECK(gguf_qwen35_index(m,err));
    for(auto& cse:cases){auto it=m.index.find(cse.name);CHECK(it!=m.index.end());if(it==m.index.end())continue;
      std::vector<float> got(cse.hf.size());bool ok=m.read_gguf_f32(it->second,got.data(),err);CHECK(ok);
      if(!ok)std::fprintf(stderr,"read %s: %s\n",cse.name.c_str(),err.c_str());
      if(!same(got,cse.hf)){size_t i=0;while(i<got.size()&&got[i]==cse.hf[i])++i;
        std::fprintf(stderr,"mismatch %s[%zu]: got %.9g want %.9g\n",cse.name.c_str(),i,
                     i<got.size()?got[i]:0,i<cse.hf.size()?cse.hf[i]:0);}
      CHECK(same(got,cse.hf));}
    // Negative control: the wrong identity inverse must fail on tiled V rows.
    std::vector<float> wrong(cases[1].hf.size());std::memcpy(wrong.data(),ts[1].raw.data(),ts[1].raw.size());
    CHECK(!same(wrong,cases[1].hf));CHECK(m.gguf_a_mismatches==0&&m.gguf_norm_mismatches==0);
    ::unlink(file.c_str());
}
static void small_loader(){
    constexpr int h=16,nk=4,nv=8,inter=4,experts=2,dk=2,dv=2;
    int c=2*nk*dk+nv*dv;
    std::vector<TS> ts;
    auto t=[&](std::string name,std::vector<uint64_t> dims,float base=0){size_t n=1;for(auto d:dims)n*=d;add(ts,name,dims,seq(n,base));};
    t("token_embd.weight",{h,4});t("output.weight",{h,4});
    add(ts,"output_norm.weight",{h},std::vector<float>(h,1.5f));
    for(int l=0;l<2;++l){std::string b="blk."+std::to_string(l)+".";
      add(ts,b+"attn_norm.weight",{h},std::vector<float>(h,1.5f));
      add(ts,b+"post_attention_norm.weight",{h},std::vector<float>(h,1.5f));
      if(!l){t(b+"attn_qkv.weight",{h,uint64_t(c)});t(b+"attn_gate.weight",{h,uint64_t(nv*dv)});
        t(b+"ssm_alpha.weight",{h,nv});t(b+"ssm_beta.weight",{h,nv});
        t(b+"ssm_conv1d.weight",{4,uint64_t(c)});
        add(ts,b+"ssm_a",{nv},std::vector<float>(nv,float(-std::exp(0.5))));
        t(b+"ssm_dt.bias",{nv});t(b+"ssm_norm.weight",{dv});t(b+"ssm_out.weight",{uint64_t(nv*dv),h});
      }else{t(b+"attn_q.weight",{h,uint64_t(2*nk*dk)});t(b+"attn_k.weight",{h,2});
        t(b+"attn_v.weight",{h,2});t(b+"attn_output.weight",{uint64_t(nk*dk),h});
        add(ts,b+"attn_q_norm.weight",{dk},std::vector<float>(dk,1.5f));
        add(ts,b+"attn_k_norm.weight",{dk},std::vector<float>(dk,1.5f));}
      t(b+"ffn_gate_inp.weight",{h,experts});t(b+"ffn_gate_inp_shexp.weight",{h});
      t(b+"ffn_gate_shexp.weight",{h,inter});t(b+"ffn_up_shexp.weight",{h,inter});
      t(b+"ffn_down_shexp.weight",{inter,h});
      t(b+"ffn_gate_exps.weight",{h,inter,experts},10);
      t(b+"ffn_up_exps.weight",{h,inter,experts},20);
      t(b+"ffn_down_exps.weight",{inter,h,experts},30);
    }
    std::string dir=tmp("model");::mkdir(dir.c_str(),0700);std::string file=dir+"/tiny.gguf";
    write(file,build_file(metadata(nk,nv,h,dk,2,experts,inter),ts));
    {std::ofstream f(dir+"/tokenizer.json");f<<"{\"model\":{\"vocab\":{\"a\":0,\"b\":1,\"c\":2,\"d\":3},\"merges\":[]}}";}
    GGUFFile f;std::string err,j;CHECK(f.open(file,err));CHECK(gguf_qwen35_config(f,"",j,err));
    auto tc=Json::parse(j).at("text_config");
    CHECK(tc.at("num_hidden_layers").integer()==2&&tc.at("vocab_size").integer()==4);
    CHECK(tc.at("layer_types").array[0].string=="linear_attention");
    CHECK(tc.at("layer_types").array[1].string=="full_attention");
    Json bad_cfg=Json::parse(j);bad_cfg.object["text_config"].object["hidden_size"].number=17;
    bad_cfg.object["text_config"].object["hidden_size"].literal.clear();
    std::string bad=bad_cfg.dump();
    bool cross=gguf_qwen35_config(f,bad,j,err);
    if(cross||err.find("hidden_size")==std::string::npos)
        std::fprintf(stderr,"crosscheck: success=%d err=%s\n",cross,err.c_str());
    CHECK(!cross&&err.find("hidden_size")!=std::string::npos);
    {std::ofstream sidecar(dir+"/config.json");sidecar<<bad;}
    Qwen35Model sidecar_bad;err.clear();CHECK(!sidecar_bad.load(dir,err));
    CHECK(err.find("hidden_size")!=std::string::npos&&
          err.find("16")!=std::string::npos&&err.find("17")!=std::string::npos);
    ::unlink((dir+"/config.json").c_str());
    Qwen35Model m;err.clear();CHECK(m.load(dir,err));if(!err.empty())std::fprintf(stderr,"loader: %s\n",err.c_str());
    CHECK(m.gguf_file&&m.layers.size()==2&&m.layers[0].la_in_qkv.gguf&&m.layers[1].q_proj.gguf);
    CHECK(m.cfg.mrope&&m.cfg.mrope_section==std::vector<int>({1,1,0}));
    CHECK(m.cfg.mtp_n_layers==1);
    CHECK(m.layers[0].e_gate_p[1].gguf_expert==1);
    std::vector<float> v(size_t(inter)*h);CHECK(m.read_gguf_f32(m.layers[0].e_gate_p[1],v.data(),err));
    CHECK(v[0]==10+float(inter*h)/8);
    CHECK(m.read_gguf_f32(m.layers[0].e_up_p[1],v.data(),err));
    CHECK(v[0]==20+float(inter*h)/8);
    CHECK(m.read_gguf_f32(m.layers[0].e_down_p[1],v.data(),err));
    CHECK(v[0]==30+float(inter*h)/8);
    std::vector<float> w(h);CHECK(m.read_gguf_f32(m.layers[0].input_norm,w.data(),err));CHECK(w[0]==0.5f);
    std::vector<float> a(nv);CHECK(m.read_gguf_f32(m.layers[0].la_A_log,a.data(),err));
    CHECK(a[0]==0.5f&&m.gguf_a_mismatches==0&&m.gguf_norm_mismatches==0);
    CHECK(m.layers[0].e_gate_p[1].gguf_half==0&&m.layers[0].e_up_p[1].gguf_half==1);
    write(dir+"/second.gguf",build_file(metadata(nk,nv,h,dk,2,experts,inter),ts));
    Qwen35Model multiple;err.clear();CHECK(!multiple.load(dir,err));
    CHECK(err.find("tiny.gguf")!=std::string::npos&&err.find("second.gguf")!=std::string::npos);
    ::unlink((dir+"/second.gguf").c_str());
    {std::ofstream f(dir+"/tokenizer.json");f<<"{\"model\":{\"vocab\":{\"a\":0,\"wrong\":1,\"c\":2,\"d\":3}}}";}
    Qwen35Model mismatch;err.clear();CHECK(!mismatch.load(dir,err));
    CHECK(err.find("token 1")!=std::string::npos);
    ::unlink((dir+"/tokenizer.json").c_str());::unlink(file.c_str());::rmdir(dir.c_str());
}
static void real_headers(){
    std::ifstream cf("tests/gguf/data/ornith_config.json");CHECK(bool(cf));if(!cf)return;
    std::string hf((std::istreambuf_iterator<char>(cf)),{});
    for(const char* path:{"/tmp/hdr-Q4_K_M.bin","/tmp/hdr-BF16.bin"}){
      std::ifstream probe(path,std::ios::binary);
      if(!probe){std::printf("SKIP real GGUF header %s (absent)\n",path);continue;}
      GGUFFile f;std::string err,j;CHECK(f.open(path,err,true));
      CHECK(gguf_qwen35_config(f,hf,j,err));if(!err.empty())std::fprintf(stderr,"%s: %s\n",path,err.c_str());
      Qwen35Model model;model.cfg.n_layers=40;model.cfg.hidden=2048;
      model.cfg.n_experts=256;model.cfg.moe_inter=512;
      model.cfg.lin_k_heads=16;model.cfg.lin_v_heads=32;
      model.cfg.lin_k_dim=128;model.cfg.lin_v_dim=128;
      model.gguf_file=std::make_unique<GGUFFile>();err.clear();
      CHECK(model.gguf_file->open(path,err,true));
      CHECK(gguf_qwen35_index(model,err));
      if(!err.empty())std::fprintf(stderr,"index %s: %s\n",path,err.c_str());
      CHECK(model.index.count("model.language_model.layers.39.mlp.experts.255.down_proj.weight")==1);
    }
}
int main(){transforms(4,8);transforms(16,32);small_loader();real_headers();
  std::printf("GGUF qwen35: %d assertions, %d failures\n",checks,failures);return failures?1:0;}
