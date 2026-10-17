// Host checks for the Unsloth Gemma 4 GGUF headers and raw UTF-8 BPE.
// Golden ids were generated with tokenizers 0.22.2 from the pinned HF JSON.
#include "b70/gguf_qwen35.hpp"
#include "b70/tokenizer.hpp"
#include "b70/json.hpp"
#include "b70/gguf_decode.hpp"
#include "gguf/test_writer.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include <unistd.h>
using namespace b70;
using namespace gguf_test_writer;
static int checks=0,failures=0;
#define CHECK(x) do{++checks;if(!(x)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);++failures;}}while(0)
static void write(const std::string& path,const Bytes& b){
    std::ofstream o(path,std::ios::binary);o.write((const char*)b.data(),b.size());
}
static void real_headers(){
    std::ifstream cf(".ref/hf-gemma-4-12b-it-config.json");
    if(!cf){std::puts("SKIP Gemma 4 reference config (absent)");return;}
    const std::string hf((std::istreambuf_iterator<char>(cf)),{});
    for(const char* path:{".ref/g4-12b-Q4_K_M.bin",".ref/g4-12b-BF16.bin"}){
        std::ifstream probe(path,std::ios::binary);
        if(!probe){std::printf("SKIP Gemma 4 header %s (absent)\n",path);continue;}
        GGUFFile f;std::string err,j;
        CHECK(f.open(path,err,true));if(!err.empty())std::fprintf(stderr,"%s\n",err.c_str());
        err.clear();CHECK(gguf_gemma4_config(f,hf,j,err));
        if(!err.empty())std::fprintf(stderr,"%s: %s\n",path,err.c_str());
        if(j.empty())continue;
        Json root=Json::parse(j),tc=root.at("text_config");
        CHECK(tc.at("model_type").string=="gemma4_unified_text");
        CHECK(tc.at("head_dim").integer()==256);
        CHECK(tc.at("global_head_dim").integer()==512);
        CHECK(tc.at("num_key_value_heads").integer()==8);
        CHECK(tc.at("num_global_key_value_heads").integer()==1);
        CHECK(tc.at("attention_k_eq_v").boolean);
        CHECK(tc.at("tie_word_embeddings").boolean);
        CHECK(tc.at("layer_types").array.size()==48);
        CHECK(tc.at("rope_parameters").at("full_attention").at("partial_rotary_factor").number==0.25);
        CHECK(tc.at("final_logit_softcapping").number==30);
        Qwen35Model m;m.cfg.n_layers=48;m.cfg.hidden=3840;m.cfg.vocab=262144;
        m.cfg.n_heads=16;m.cfg.n_kv_heads=8;m.cfg.n_global_kv_heads=1;
        m.cfg.head_dim=256;m.cfg.global_head_dim=512;m.cfg.dense_inter=15360;
        for(const auto& layer:tc.at("layer_types").array){
            m.cfg.muse_sliding_attention.push_back(layer.string=="sliding_attention");
            m.cfg.layer_types.push_back(LayerKind::FULL_ATTN);
        }
        m.gguf_file=std::make_unique<GGUFFile>();
        err.clear();CHECK(m.gguf_file->open(path,err,true));
        CHECK(gguf_gemma4_index(m,err));
        if(!err.empty())std::fprintf(stderr,"index %s: %s\n",path,err.c_str());
        CHECK(m.index.size()==666); // every one of 667 tensors except consumed rope_freqs
        CHECK(m.index.count("model.language_model.layers.5.layer_scalar")==1);
        CHECK(m.index.count("model.language_model.layers.5.self_attn.v_proj.weight")==0);
        CHECK(m.index.count("model.language_model.layers.4.self_attn.v_proj.weight")==1);
        CHECK(m.index.count("model.language_model.norm.weight")==1);
        CHECK(m.index.count("lm_head.weight")==0);
        Json bad=Json::parse(hf);
        bad.object["text_config"].object["head_dim"].number=128;
        bad.object["text_config"].object["head_dim"].literal.clear();
        err.clear();CHECK(!gguf_gemma4_config(f,bad.dump(),j,err));
        CHECK(err.find("head_dim")!=std::string::npos);
    }
}
static void synthetic(){
    const std::string path="/tmp/gguf-gemma4-"+std::to_string(getpid())+".gguf";
    const auto* q4=gguf_type_info(12);const auto* q6=gguf_type_info(14);
    CHECK(q4&&q6);if(!q4||!q6)return;
    Bytes scale(4);float w=0.75f;std::memcpy(scale.data(),&w,4);
    std::vector<TS> ts={
        {"blk.0.attn_q.weight",{256,1},12,0,Bytes(q4->block_bytes,0)},
        {"blk.0.ffn_down.weight",{256,1},14,256,Bytes(q6->block_bytes,0)},
        {"blk.0.layer_output_scale.weight",{1},0,512,scale}};
    const std::vector<KV> meta={{"general.alignment",4,num(64,4)}};
    write(path,build_file(meta,ts));
    Qwen35Model m;m.cfg.n_layers=1;m.gguf_file=std::make_unique<GGUFFile>();
    std::string err;CHECK(m.gguf_file->open(path,err));
    CHECK(gguf_gemma4_index(m,err));
    auto q=m.index.at("model.language_model.layers.0.self_attn.q_proj.weight");
    auto d=m.index.at("model.language_model.layers.0.mlp.down_proj.weight");
    CHECK(q.gguf->type==12&&d.gguf->type==14);
    std::vector<uint8_t> raw(q4->block_bytes,0xff);
    CHECK(m.read_gguf_raw_rows(q,raw.data(),err));
    CHECK(raw==Bytes(q4->block_bytes,0));
    std::vector<uint8_t> raw6(q6->block_bytes,0xff);
    CHECK(m.read_gguf_raw_rows(d,raw6.data(),err));
    CHECK(raw6==Bytes(q6->block_bytes,0));
    float got=0;CHECK(m.read_gguf_f32(m.index.at("model.language_model.layers.0.layer_scalar"),
                                      &got,err));
    CHECK(got==w);
    m.gguf_file->close();::unlink(path.c_str());
    ts.push_back({"blk.0.vision.weight",{1},0,576,scale});
    write(path,build_file(meta,ts));err.clear();
    CHECK(m.gguf_file->open(path,err));
    CHECK(!gguf_gemma4_index(m,err));
    CHECK(err.find("blk.0.vision.weight")!=std::string::npos);
    m.gguf_file->close();::unlink(path.c_str());
}
static void tokenizer_fallback(){
    const std::string path="/tmp/gguf-gemma4-tokenizer-"+std::to_string(getpid())+".gguf";
    const std::vector<std::string> tokens={"<pad>","<eos>","<bos>","<unk>","a","b","ab",
        "▁","▁a","<0xC3>","<0xA9>","<tag>"};
    std::vector<KV> kv={kv_string("tokenizer.ggml.model","gemma4"),
        kv_array_strings("tokenizer.ggml.tokens",tokens),
        kv_array_i32("tokenizer.ggml.token_type",{3,3,3,2,1,1,1,1,1,6,6,4}),
        kv_array_f32("tokenizer.ggml.scores",std::vector<float>(tokens.size(),0)),
        kv_array_strings("tokenizer.ggml.merges",{"a b","▁ a"}),
        {"tokenizer.ggml.bos_token_id",4,num(2,4)},
        {"tokenizer.ggml.eos_token_id",4,num(1,4)},
        {"tokenizer.ggml.unknown_token_id",4,num(3,4)},
        {"tokenizer.ggml.add_bos_token",7,num(1,1)},
        {"tokenizer.ggml.add_space_prefix",7,num(0,1)}};
    write(path,build_file(kv,{}));
    GGUFFile f;Tokenizer tk;std::string err;
    CHECK(f.open(path,err));CHECK(tk.load_gguf(f,err));
    CHECK(tk.encode("ab a",true)==std::vector<int32_t>({2,6,8}));
    CHECK(tk.encode("é")==std::vector<int32_t>({9,10}));
    CHECK(tk.encode("<tag>a")==std::vector<int32_t>({11,4}));
    CHECK(tk.decode({6,8})=="ab a");
    CHECK(tk.decode({9,10})=="é");
    CHECK(tk.bos()==2&&tk.eos()==1&&tk.add_bos_token());
    f.close();
    for(auto& x:kv)if(x.key=="tokenizer.ggml.add_bos_token")x.payload=num(0,1);
    write(path,build_file(kv,{}));err.clear();
    CHECK(f.open(path,err));CHECK(tk.load_gguf(f,err));
    CHECK(tk.encode("ab",true)==std::vector<int32_t>({2,6}));
    f.close();
    for(auto& x:kv)if(x.key=="tokenizer.ggml.add_space_prefix")x.payload=num(1,1);
    write(path,build_file(kv,{}));err.clear();
    CHECK(f.open(path,err));CHECK(!tk.load_gguf(f,err));
    CHECK(err.find("add_space_prefix")!=std::string::npos);
    f.close();::unlink(path.c_str());
}
static void bare_loader(){
    const std::string path="/tmp/gguf-gemma4-bare-"+std::to_string(getpid())+".gguf";
    auto u=[](std::string k,uint32_t x){return KV{k,4,num(x,4)};};
    auto fl=[](std::string k,float x){uint32_t bits;std::memcpy(&bits,&x,4);return KV{k,6,num(bits,4)};};
    auto ua=[](std::string k,std::vector<uint32_t> x){
        Bytes b;put(b,4,4);put(b,x.size(),8);for(auto v:x)put(b,v,4);return KV{k,9,b};
    };
    auto ba=[](std::string k,std::vector<uint8_t> x){
        Bytes b;put(b,7,4);put(b,x.size(),8);for(auto v:x)put(b,v,1);return KV{k,9,b};
    };
    const std::string p="gemma4.";
    std::vector<KV> meta={u("general.alignment",64),kv_string("general.architecture","gemma4"),
        u(p+"block_count",2),u(p+"embedding_length",16),u(p+"feed_forward_length",32),
        u(p+"context_length",256),u(p+"attention.head_count",1),
        ua(p+"attention.head_count_kv",{1,1}),
        ba(p+"attention.sliding_window_pattern",{1,0}),
        u(p+"attention.key_length",32),u(p+"attention.key_length_swa",16),
        u(p+"attention.value_length",32),u(p+"attention.value_length_swa",16),
        u(p+"rope.dimension_count",32),u(p+"rope.dimension_count_swa",16),
        u(p+"attention.sliding_window",64),u(p+"attention.shared_kv_layers",0),
        u(p+"embedding_length_per_layer_input",0),
        fl(p+"rope.freq_base",1000000),fl(p+"rope.freq_base_swa",10000),
        fl(p+"attention.layer_norm_rms_epsilon",1e-6f),
        fl(p+"final_logit_softcapping",30),
        kv_string("tokenizer.ggml.model","gemma4"),
        kv_array_strings("tokenizer.ggml.tokens",{"<pad>","<eos>","<bos>","<unk>","a"}),
        kv_array_i32("tokenizer.ggml.token_type",{3,3,3,2,1}),
        kv_array_f32("tokenizer.ggml.scores",{0,0,0,0,0}),
        kv_array_strings("tokenizer.ggml.merges",{}),
        u("tokenizer.ggml.bos_token_id",2),u("tokenizer.ggml.eos_token_id",1),
        u("tokenizer.ggml.unknown_token_id",3),
        KV{"tokenizer.ggml.add_bos_token",7,num(1,1)},
        KV{"tokenizer.ggml.add_space_prefix",7,num(0,1)}};
    Bytes emb(16*4*4),value(16*16*4),freq(16*4);
    for(int i=0;i<16;++i){float x=i<4?1.0f:1e30f;std::memcpy(freq.data()+4*i,&x,4);}
    std::vector<TS> ts={{"token_embd.weight",{16,4},0,0,emb},
                        {"blk.0.attn_v.weight",{16,16},0,256,value},
                        {"rope_freqs.weight",{16},0,1280,freq}};
    write(path,build_file(meta,ts));
    Qwen35Model model;std::string err;
    CHECK(model.load(path,err,false,true));if(!err.empty())std::fprintf(stderr,"bare loader: %s\n",err.c_str());
    CHECK(model.cfg.is_gemma4&&model.cfg.n_layers==2);
    CHECK(model.cfg.head_dim==16&&model.cfg.global_head_dim==32);
    CHECK(model.cfg.global_partial_rope==0.25f);
    CHECK(model.cfg.tie_embeddings&&model.cfg.k_eq_v);
    CHECK(model.index.count("model.language_model.embed_tokens.weight")==1);
    auto changed=meta;
    for(auto& x:changed)if(x.key==p+"attention.shared_kv_layers")x=u(x.key,1);
    write(path,build_file(changed,ts));err.clear();
    CHECK(!model.load(path,err,false,true));
    CHECK(err.find("attention.shared_kv_layers")!=std::string::npos);
    changed=meta;
    for(auto& x:changed)if(x.key==p+"embedding_length_per_layer_input")x=u(x.key,8);
    write(path,build_file(changed,ts));err.clear();
    CHECK(!model.load(path,err,false,true));
    CHECK(err.find("embedding_length_per_layer_input")!=std::string::npos);
    ::unlink(path.c_str());
}
static void tokenizer_real(){
    std::ifstream probe(".ref/g4-12b-Q4_K_M.bin",std::ios::binary);
    std::ifstream tprobe(".ref/hf-gemma-4-12b-it-tokenizer.json");
    if(!probe||!tprobe){std::puts("SKIP Gemma 4 tokenizer reference (absent)");return;}
    char cwd[4096];CHECK(::getcwd(cwd,sizeof cwd)!=nullptr);
    const std::string dir="/tmp/gguf-gemma4-hf-"+std::to_string(getpid());
    ::mkdir(dir.c_str(),0700);
    const std::string tj=dir+"/tokenizer.json",tc=dir+"/tokenizer_config.json";
    ::symlink((std::string(cwd)+"/.ref/hf-gemma-4-12b-it-tokenizer.json").c_str(),tj.c_str());
    ::symlink((std::string(cwd)+"/.ref/hf-gemma-4-12b-it-tokenizer_config.json").c_str(),tc.c_str());
    Tokenizer hf,gg;GGUFFile f;std::string err;
    CHECK(hf.load(dir,err));if(!err.empty())std::fprintf(stderr,"HF tokenizer: %s\n",err.c_str());
    err.clear();CHECK(f.open(".ref/g4-12b-Q4_K_M.bin",err,true));
    CHECK(gg.load_gguf(f,err));if(!err.empty())std::fprintf(stderr,"GGUF tokenizer: %s\n",err.c_str());
    CHECK(hf.vocab_size()==gg.vocab_size());
    CHECK(hf.merge_count()==gg.merge_count());
    CHECK(hf.bos()==gg.bos()&&hf.eos()==gg.eos()&&hf.eos2()==gg.eos2());
    const std::string chat="<|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
    CHECK(gg.apply_chat_template("Hello")==chat);
    CHECK(hf.apply_chat_template("Hello")==chat);
    CHECK(gg.encode(chat,true)==hf.encode(chat,true));
    CHECK(!gg.encode(chat,true).empty()&&gg.encode(chat,true).front()==2);
    ChatOptions thinking;thinking.enable_thinking=true;
    CHECK(gg.apply_chat_template(std::vector<ChatMessage>{{"user","Hello","",false,{}}},thinking)==
          "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n");
    CHECK(gg.bos()==2&&gg.eos()==106&&gg.eos2()==1&&gg.unk()==3&&
          gg.pad()==0&&gg.mask()==4&&gg.add_bos_token());
    CHECK(gg.suppress_tokens()==std::vector<int32_t>({258883,258882}));
    struct Golden{const char* text;std::vector<int32_t> ids;};
    const std::vector<Golden> gold={
        {"Hello, world!",{9259,236764,1902,236888}},
        {"  spaces   and\nnewlines\n\n",{138,35220,139,624,107,208697,108}},
        {"שלום עולם",{82110,42737,11173,87859}},
        {"こんにちは世界",{85141,12811}},
        {"def f(x):\n    return x + 1",{2063,517,236769,236781,1473,107,140,2060,1123,900,236743,236770}},
        {"a\tb  c",{236746,255968,236763,138,236755}},
        {"😀 café é",{242398,33443,545,238288}},
        {"<|tool_call>call:foo{}<tool_call|>",{48,6639,236787,26537,16454,49}}};
    for(const auto& g:gold){
        auto a=hf.encode(g.text),b=gg.encode(g.text);
        if(a!=g.ids||b!=g.ids)std::fprintf(stderr,"Gemma tokenizer mismatch: %s (HF %zu, GGUF %zu, golden %zu)\n",
                                            g.text,a.size(),b.size(),g.ids.size());
        CHECK(a==g.ids);CHECK(b==g.ids);CHECK(a==b);
        CHECK(hf.encode(g.text,true)==gg.encode(g.text,true));
    }
    f.close();::unlink(tj.c_str());::unlink(tc.c_str());::rmdir(dir.c_str());
}
int main(){
    real_headers();synthetic();tokenizer_fallback();bare_loader();tokenizer_real();
    std::printf("GGUF gemma4: %d checks, %d failures\n",checks,failures);
    return failures?1:0;
}
