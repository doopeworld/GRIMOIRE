// Host checks for standard Qwen3 GGUF, using real published header slices.
#include "b70/gguf_qwen35.hpp"
#include "b70/tokenizer.hpp"
#include "b70/json.hpp"
#include "b70/moe.hpp"
#include "gguf/test_writer.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
using namespace b70;
using namespace gguf_test_writer;
static int checks=0, failures=0;
#define CHECK(x) do { ++checks; if(!(x)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);++failures;} }while(0)
static std::string read(const char* path){std::ifstream f(path);return {std::istreambuf_iterator<char>(f),{}};}
static void header(const char* path,const char* hf_path,bool moe){
    GGUFFile f;std::string err,j;
    CHECK(f.open(path,err,true));if(!err.empty()){std::fprintf(stderr,"%s\n",err.c_str());return;}
    CHECK(gguf_qwen3_config(f,read(hf_path),j,err));
    if(j.empty()){std::fprintf(stderr,"config: %s\n",err.c_str());return;}
    Json c=Json::parse(j),hf=Json::parse(read(hf_path));
    CHECK(c.at("model_type").string==(moe?"qwen3_moe":"qwen3"));
    for(const char* key:{"num_hidden_layers","hidden_size","vocab_size",
                         "num_attention_heads","num_key_value_heads","head_dim",
                         "rope_theta","rms_norm_eps","max_position_embeddings"})
        CHECK(std::abs(c.at(key).number-hf.at(key).number)<=
              1e-6*std::max(1e-12,std::abs(hf.at(key).number)));
    CHECK(c.at("partial_rotary_factor").number==1);
    CHECK(c.at("attn_output_gate").boolean==false);
    CHECK(c.at("tie_word_embeddings").boolean==false);
    CHECK(c.at("head_dim").integer()==128);
    if(moe){
        for(const char* key:{"num_experts","num_experts_per_tok","moe_intermediate_size",
                             "decoder_sparse_step"})CHECK(c.at(key).number==hf.at(key).number);
        CHECK(c.at("norm_topk_prob").boolean);
        CHECK(c.at("mlp_only_layers").array.empty());
        // Published GGUF says 5472; HF says 6144. There is no dense MLP
        // in this 48-layer file, so the field is never used for execution.
        CHECK(c.at("intermediate_size").integer()==5472);
        CHECK(hf.at("intermediate_size").integer()==6144);
    } else CHECK(c.at("intermediate_size").number==hf.at("intermediate_size").number);
    Qwen35Model m;
    m.cfg.is_qwen3=true;m.cfg.n_layers=c.at("num_hidden_layers").integer();
    m.cfg.hidden=c.at("hidden_size").integer();m.cfg.vocab=c.at("vocab_size").integer();
    m.cfg.n_heads=c.at("num_attention_heads").integer();
    m.cfg.n_kv_heads=c.at("num_key_value_heads").integer();
    m.cfg.head_dim=c.at("head_dim").integer();
    m.cfg.dense_inter=c.at("intermediate_size").integer();
    m.cfg.rope_theta=float(c.at("rope_theta").number);
    m.cfg.partial_rope=float(c.at("partial_rotary_factor").number);
    if(moe){m.cfg.n_experts=c.at("num_experts").integer();
        m.cfg.top_k=c.at("num_experts_per_tok").integer();
        m.cfg.moe_inter=c.at("moe_intermediate_size").integer();}
    CHECK(m.cfg.norm_weight_offset()==0);
    CHECK(std::abs(m.cfg.attn_softmax_scale(128)-1/std::sqrt(128.f))<1e-7f);
    m.gguf_file=std::make_unique<GGUFFile>();
    CHECK(m.gguf_file->open(path,err,true));
    CHECK(gguf_qwen3_index(m,err));
    if(!err.empty())std::fprintf(stderr,"index: %s\n",err.c_str());
    CHECK(m.index.count("model.layers.0.self_attn.q_norm.weight")==1);
    CHECK(m.index.count("model.layers.0.self_attn.k_norm.weight")==1);
    CHECK(m.index.count("model.layers.0.post_attention_layernorm.weight")==1);
    CHECK(m.index.at("model.layers.0.self_attn.q_norm.weight").gguf_inverse==TensorRef::GGUFInverse::NONE);
    CHECK(m.index.count("lm_head.weight")==1);
    if(moe){
        CHECK(m.index.count("model.layers.47.mlp.experts.127.gate_proj.weight")==1);
        CHECK(m.index.count("model.layers.47.mlp.experts.127.up_proj.weight")==1);
        CHECK(m.index.count("model.layers.47.mlp.experts.127.down_proj.weight")==1);
        CHECK(m.index.count("model.layers.0.mlp.shared_expert.gate_proj.weight")==0);
    }else CHECK(m.index.count("model.layers.35.mlp.gate_proj.weight")==1);
    Tokenizer tok;err.clear();CHECK(tok.load_gguf(f,err));
    if(!err.empty())std::fprintf(stderr,"tokenizer: %s\n",err.c_str());
    CHECK(tok.vocab_size()==151936);
    CHECK(tok.eos()==151645);
    if(const char* dir=std::getenv("GRIMOIRE_QWEN3_HF_TOKENIZER_DIR")){
        Tokenizer hf_tok;std::string why;
        CHECK(hf_tok.load(dir,why));
        // GGUF pads the published HF vocabulary to the embedding row count.
        CHECK(hf_tok.vocab_size()<=tok.vocab_size());
        CHECK(hf_tok.merge_count()==tok.merge_count());
        if(hf_tok.vocab_size()<=tok.vocab_size())
            for(size_t id=0;id<hf_tok.vocab_size();++id)
                CHECK(tok.raw_token(id)==hf_tok.raw_token(id));
    }
    const Json gold=Json::parse(read("tests/tokenizer_golden/qwen3_hf_golden.json"));
    for(const Json& c:gold.at("cases").array){
        std::vector<int32_t> want;
        for(const Json& id:c.at("ids").array)want.push_back(id.integer());
        const auto got=tok.encode(c.at("text").string);
        if(got!=want)std::fprintf(stderr,"tokenizer mismatch on %s\n",c.at("text").dump().c_str());
        CHECK(got==want);
    }
}
static void plain_weights(){
    std::vector<gguf_test_writer::TS> ts;
    auto tensor=[&](const std::string& name,std::vector<uint64_t> shape,std::vector<float> values){
        gguf_test_writer::Bytes bytes(values.size()*4);
        std::memcpy(bytes.data(),values.data(),bytes.size());
        uint64_t off=ts.empty()?0:(ts.back().offset+ts.back().raw.size()+31)&~uint64_t(31);
        ts.push_back({name,shape,0,off,std::move(bytes)});
    };
    tensor("blk.0.attn_q_norm.weight",{4},{0.5f,1.f,1.5f,2.f});
    tensor("blk.0.attn_q.weight",{4,4},{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16});
    const std::string path="/tmp/grimoire-qwen3-plain-"+std::to_string(getpid())+".gguf";
    auto bytes=build_file({kv_string("general.architecture","qwen3")},ts);
    {std::ofstream o(path,std::ios::binary);o.write((const char*)bytes.data(),bytes.size());}
    Qwen35Model m;m.cfg.is_qwen3=true;m.cfg.n_layers=1;
    m.gguf_file=std::make_unique<GGUFFile>();std::string err;
    CHECK(m.gguf_file->open(path,err));CHECK(gguf_qwen3_index(m,err));
    float norm[4]={};CHECK(m.read_gguf_f32(m.index.at("model.layers.0.self_attn.q_norm.weight"),norm,err));
    for(int i=0;i<4;++i)CHECK(norm[i]==0.5f+0.5f*i);
    // Qwen3RMSNorm.forward: weight * (x / sqrt(mean(x²)+eps)).
    const float x[4]={1,2,3,4}, mean=7.5f, eps=1e-6f;
    const float expected=x[0]/std::sqrt(mean+eps)*0.5f;
    const float got=x[0]/std::sqrt(mean+eps)*(m.cfg.norm_weight_offset()+norm[0]);
    CHECK(std::abs(got-expected)<1e-7f);
    CHECK(std::abs(got-x[0]/std::sqrt(mean+eps)*1.5f)>0.1f); // wrong (1+w) control
    float q[16]={};CHECK(m.read_gguf_f32(m.index.at("model.layers.0.self_attn.q_proj.weight"),q,err));
    for(int i=0;i<16;++i)CHECK(q[i]==float(i+1)); // no LLAMA Q permutation
    ::unlink(path.c_str());
}
static void tied_embeddings(){
    auto u32=[](std::string key,uint32_t v){return KV{std::move(key),4,num(v,4)};};
    auto f32=[](std::string key,float v){uint32_t bits;std::memcpy(&bits,&v,4);
        return KV{std::move(key),6,num(bits,4)};};
    std::vector<KV> kv={kv_string("general.architecture","qwen3"),
        u32("qwen3.block_count",1),u32("qwen3.embedding_length",4),
        u32("qwen3.feed_forward_length",8),u32("qwen3.attention.head_count",2),
        u32("qwen3.attention.head_count_kv",1),u32("qwen3.attention.key_length",2),
        u32("qwen3.attention.value_length",2),u32("qwen3.context_length",16),
        f32("qwen3.rope.freq_base",1e6f),
        f32("qwen3.attention.layer_norm_rms_epsilon",1e-6f)};
    std::vector<TS> tensors={{"token_embd.weight",{4,6},0,0,Bytes(4*6*4)},
        {"blk.0.attn_q.weight",{4,4},0,128,Bytes(4*4*4)}};
    const std::string path="/tmp/grimoire-qwen3-tied-"+std::to_string(getpid())+".gguf";
    const auto bytes=build_file(kv,tensors);
    {std::ofstream out(path,std::ios::binary);out.write((const char*)bytes.data(),bytes.size());}
    GGUFFile f;std::string err,j;CHECK(f.open(path,err,true));
    CHECK(gguf_qwen3_config(f,"",j,err));
    if(!j.empty())CHECK(Json::parse(j).at("tie_word_embeddings").boolean);
    Qwen35Model m;m.cfg.is_qwen3=true;m.cfg.n_layers=1;
    m.gguf_file=std::make_unique<GGUFFile>();CHECK(m.gguf_file->open(path,err,true));
    CHECK(gguf_qwen3_index(m,err));
    CHECK(m.index.count("model.embed_tokens.weight")==1);
    CHECK(m.index.count("lm_head.weight")==0);
    ::unlink(path.c_str());
}
static void mixed_moe_pattern(){
    auto u32=[](std::string key,uint32_t v){return KV{std::move(key),4,num(v,4)};};
    auto f32=[](std::string key,float v){uint32_t bits;std::memcpy(&bits,&v,4);
        return KV{std::move(key),6,num(bits,4)};};
    std::vector<KV> kv={kv_string("general.architecture","qwen3moe"),
        u32("qwen3moe.block_count",2),u32("qwen3moe.embedding_length",4),
        u32("qwen3moe.feed_forward_length",8),u32("qwen3moe.attention.head_count",2),
        u32("qwen3moe.attention.head_count_kv",1),
        u32("qwen3moe.attention.key_length",2),u32("qwen3moe.attention.value_length",2),
        u32("qwen3moe.context_length",16),u32("qwen3moe.expert_count",2),
        u32("qwen3moe.expert_used_count",1),u32("qwen3moe.expert_feed_forward_length",2),
        f32("qwen3moe.rope.freq_base",1e6f),
        f32("qwen3moe.attention.layer_norm_rms_epsilon",1e-6f)};
    std::vector<TS> tensors;
    uint64_t offset=0;
    auto add=[&](std::string name,std::vector<uint64_t> dims){
        uint64_t n=1;for(uint64_t d:dims)n*=d;
        tensors.push_back({std::move(name),dims,0,offset,Bytes(size_t(n)*4)});
        offset=(offset+n*4+31)&~uint64_t(31);
    };
    add("token_embd.weight",{4,6});
    add("blk.0.attn_q.weight",{4,4});add("blk.1.attn_q.weight",{4,4});
    for(const char* role:{"ffn_gate","ffn_up","ffn_down"})
        add(std::string("blk.0.")+role+".weight",{role==std::string("ffn_down")?8u:4u,
                                                   role==std::string("ffn_down")?4u:8u});
    add("blk.1.ffn_gate_inp.weight",{4,2});
    add("blk.1.ffn_gate_exps.weight",{4,2,2});
    add("blk.1.ffn_up_exps.weight",{4,2,2});
    add("blk.1.ffn_down_exps.weight",{2,4,2});
    const std::string path="/tmp/grimoire-qwen3-mixed-"+std::to_string(getpid())+".gguf";
    const auto bytes=build_file(kv,tensors);
    {std::ofstream out(path,std::ios::binary);out.write((const char*)bytes.data(),bytes.size());}
    GGUFFile f;std::string err,j;CHECK(f.open(path,err,true));
    CHECK(gguf_qwen3_config(f,"",j,err));
    if(!j.empty()){
        Json c=Json::parse(j);
        CHECK(c.at("mlp_only_layers").array.size()==1);
        CHECK(c.at("mlp_only_layers").array[0].integer()==0);
        // The HF step-2 rule yields the same [dense, routed] pattern.
        c.object["decoder_sparse_step"]=Json::parse("2");
        c.object["mlp_only_layers"]=Json::parse("[]");
        std::string equivalent;err.clear();
        CHECK(gguf_qwen3_config(f,c.dump(),equivalent,err));
    }
    Qwen35Model m;m.cfg.is_qwen3=true;m.cfg.n_layers=2;m.cfg.n_experts=2;
    m.cfg.hidden=4;m.cfg.dense_inter=8;m.cfg.moe_inter=2;m.cfg.mlp_only_layers={0};
    m.gguf_file=std::make_unique<GGUFFile>();CHECK(m.gguf_file->open(path,err,true));
    CHECK(gguf_qwen3_index(m,err));
    CHECK(m.index.count("model.layers.0.mlp.gate_proj.weight")==1);
    CHECK(m.index.count("model.layers.1.mlp.gate.weight")==1);
    CHECK(m.index.count("model.layers.1.mlp.experts.1.down_proj.weight")==1);
    ::unlink(path.c_str());
}
static void router_formula(){
    // modeling_qwen3_moe.py:259-263: softmax over ALL 128 logits, take
    // top eight, then divide by their sum. Logits log(1)..log(8)
    // therefore yield selected weights 8/36, 7/36, ... 1/36.
    float logits[128];for(float& x:logits)x=-100.f;
    for(int i=0;i<8;++i)logits[i]=std::log(float(i+1));
    Routing r{};route_ref(logits,128,8,true,r);
    CHECK(r.k==8);
    float sum=0;
    for(int j=0;j<8;++j){
        CHECK(r.expert[j]==7-j);
        CHECK(std::abs(r.weight[j]-float(8-j)/36.f)<1e-6f);
        sum+=r.weight[j];
    }
    CHECK(std::abs(sum-1.f)<1e-6f);
}
int main(){
    header(".ref/q3-8b-Q4_K_M.bin",".ref/qwen3-8b-config.json",false);
    header(".ref/q3c-30b-Q4_K_M.bin",".ref/qwen3-coder-30b-config.json",true);
    plain_weights();
    tied_embeddings();
    mixed_moe_pattern();
    router_formula();
    std::printf("qwen3 GGUF: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
