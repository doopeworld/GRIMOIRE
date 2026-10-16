// GRIMOIRE -- qwen35moe GGUF source adapter. SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf_qwen35.hpp"
#include "b70/gguf_decode.hpp"
#include "b70/json.hpp"
#include "b70/formats.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace b70 {
namespace {
Json num(double n) { Json j; j.kind=Json::Kind::Number; j.number=n; return j; }
Json flag(bool b) { Json j; j.kind=Json::Kind::Boolean; j.boolean=b; return j; }
Json str(const std::string& s) { Json j; j.kind=Json::Kind::String; j.string=s; return j; }
Json obj() { Json j; j.kind=Json::Kind::Object; return j; }
Json arr() { Json j; j.kind=Json::Kind::Array; return j; }
void put(Json& j,const std::string& k,Json v) { j.keys.push_back(k); j.object.emplace(k,std::move(v)); }
bool equal_json(const Json& a,const Json& b) {
    if(a.kind!=b.kind) return false;
    switch(a.kind) {
    case Json::Kind::Null: return true;
    case Json::Kind::Boolean: return a.boolean==b.boolean;
    case Json::Kind::Number:
        return std::abs(a.number-b.number)<=1e-6*std::max({1e-12,std::abs(a.number),std::abs(b.number)});
    case Json::Kind::String: return a.string==b.string;
    case Json::Kind::Array:
        if(a.array.size()!=b.array.size()) return false;
        for(size_t i=0;i<a.array.size();++i) if(!equal_json(a.array[i],b.array[i])) return false;
        return true;
    case Json::Kind::Object:
        if(a.object.size()!=b.object.size()) return false;
        for(const auto& kv:a.object) { auto* v=b.find(kv.first); if(!v||!equal_json(kv.second,*v))return false; }
        return true;
    }
    return false;
}
bool array_u32(const GGUFFile& f,const char* key,std::vector<uint32_t>& v,std::string& err) {
    const GGUFValue* a=nullptr;
    if(!f.get_array(key,a,err)) return false;
    if(auto* p=std::get_if<std::vector<uint32_t>>(&a->data)){v=*p;return true;}
    if(auto* p=std::get_if<std::vector<int32_t>>(&a->data)){
        v.clear();for(int32_t x:*p){if(x<0){err="negative GGUF dimension section";return false;}v.push_back(uint32_t(x));}
        return true;
    }
    err=std::string("GGUF array is not u32/i32: ")+key;return false;
}
}

uint64_t gguf_v_tiled_index(uint64_t grouped,uint64_t nk,uint64_t nv) {
    const uint64_t r=nv/nk;
    return (grouped%r)*nk+grouped/r;
}

bool gguf_a_log_recovery_ok(float a, float recovered) {
    if (!(a < 0.0f) || !std::isfinite(a) || !std::isfinite(recovered)) return false;
    const bf16_t rounded = f32_to_bf16(recovered);
    if (bf16_to_f32(rounded) != recovered) return false;
    const uint16_t bits = rounded.bits;
    const auto adjacent = [&](bool up) {
        // BF16's encoding is monotone within each sign, with the direction
        // reversed for negative values. Handle both signed zero encodings.
        if ((bits & 0x7fff) == 0)
            return bf16_to_f32(bf16_t{uint16_t(up ? 0x0001 : 0x8001)});
        return bf16_to_f32(bf16_t{uint16_t(bits + ((bool(bits & 0x8000) == up) ? -1 : 1))});
    };
    const double target = double(a);
    // -exp(x) - A == -(expm1(x) + (A + 1)). The latter retains the
    // separation of BF16 neighbours around x=0, where -exp(x) rounds to
    // exactly -1 in double even for a nonzero BF16 subnormal.
    const auto distance = [&](double x) { return std::abs(std::expm1(x) + (target + 1.0)); };
    const double delta = distance(double(recovered));
    const double ulp = double(std::nextafter(a, -INFINITY)) - target;
    return delta <= 4.0 * std::abs(ulp) &&
           delta < distance(double(adjacent(false))) &&
           delta < distance(double(adjacent(true)));
}

bool gguf_qwen35_config(const GGUFFile& f,const std::string& sidecar,
                         std::string& out,std::string& err) {
    std::string arch;
    if(!f.get_string("general.architecture",arch,err)) return false;
    if(arch!="qwen35moe") {err="unsupported GGUF architecture: "+arch;return false;}
    const std::string p="qwen35moe.";
    auto get=[&](const char* key,uint32_t& v){return f.get_u32(p+key,v,err);};
    uint32_t blocks,mtp,h,heads,kv,hd,vd,rope,ne,top,inter,shared,nk,nv,dk,inner,conv,interval;
    if(!get("block_count",blocks)||!get("nextn_predict_layers",mtp)||mtp>=blocks||
       !get("embedding_length",h)||!get("attention.head_count",heads)||
       !get("attention.head_count_kv",kv)||!get("attention.key_length",hd)||
       !get("attention.value_length",vd)||!get("rope.dimension_count",rope)||
       !get("expert_count",ne)||!get("expert_used_count",top)||
       !get("expert_feed_forward_length",inter)||
       !get("expert_shared_feed_forward_length",shared)||
       !get("ssm.group_count",nk)||!get("ssm.time_step_rank",nv)||
       !get("ssm.state_size",dk)||!get("ssm.inner_size",inner)||
       !get("ssm.conv_kernel",conv)||!get("full_attention_interval",interval))return false;
    float theta,eps;
    if(!f.get_f32(p+"rope.freq_base",theta,err)||
       !f.get_f32(p+"attention.layer_norm_rms_epsilon",eps,err)) return false;
    if(!hd||hd!=vd||!nk||!nv||nv%nk||inner%nv||!interval||!h||!heads||!kv) {
        err="inconsistent qwen35moe GGUF geometry";return false;
    }
    std::vector<uint32_t> sections;
    if(!array_u32(f,"qwen35moe.rope.dimension_sections",sections,err))return false;
    if(sections.size()<3){err="GGUF mrope dimension_sections needs three values";return false;}
    const GGUFTensor* emb=f.find("token_embd.weight");
    if(!emb||emb->n_dims!=2||emb->ne[0]!=h){err="GGUF token_embd.weight shape disagrees with embedding_length";return false;}
    const bool tied=!f.find("output.weight");
    bool gate=false,seen=false;
    for(uint32_t i=0;i<blocks-mtp;++i) {
        const GGUFTensor* q=f.find("blk."+std::to_string(i)+".attn_q.weight");
        if(q) {
            if(q->n_dims!=2||q->ne[0]!=h||
               (q->ne[1]!=uint64_t(heads)*hd&&q->ne[1]!=uint64_t(2)*heads*hd)){
                err="GGUF attn_q.weight has invalid query width";return false;
            }
            const bool this_gate=q->ne[1]==uint64_t(2)*heads*hd;
            if(seen&&gate!=this_gate){err="GGUF attn_q output gate differs across layers";return false;}
            gate=this_gate;seen=true;
        }
    }
    if(!seen){err="GGUF has no full-attention attn_q.weight";return false;}
    Json root=obj(),tc=obj(),rp=obj(),ms=arr(),layers=arr();
    const auto n=[&](const char* k,double v){put(tc,k,num(v));};
    n("num_hidden_layers",blocks-mtp);n("hidden_size",h);n("vocab_size",emb->ne[1]);
    n("num_attention_heads",heads);n("num_key_value_heads",kv);n("head_dim",hd);
    n("rope_theta",theta);n("partial_rotary_factor",double(rope)/hd);n("rms_norm_eps",eps);
    n("num_experts",ne);n("num_experts_per_tok",top);n("moe_intermediate_size",inter);
    n("shared_expert_intermediate_size",shared);n("linear_num_key_heads",nk);
    n("linear_num_value_heads",nv);n("linear_key_head_dim",dk);
    n("linear_value_head_dim",inner/nv);n("linear_conv_kernel_dim",conv);
    n("mtp_num_hidden_layers",mtp);n("full_attention_interval",interval);
    put(tc,"attn_output_gate",flag(gate));put(tc,"tie_word_embeddings",flag(tied));
    for(int i=0;i<3;++i)ms.array.push_back(num(sections[size_t(i)]));
    put(rp,"mrope_section",std::move(ms));put(rp,"rope_theta",num(theta));
    put(rp,"partial_rotary_factor",num(double(rope)/hd));
    put(tc,"rope_parameters",std::move(rp));
    const GGUFValue* recurrent=f.get(p+"recurrent_layers");
    const std::vector<uint8_t>* rv=nullptr;
    if(recurrent) {
        rv=std::get_if<std::vector<uint8_t>>(&recurrent->data);
        if(!rv||rv->size()!=blocks-mtp){err="GGUF recurrent_layers must be a bool array of transformer-layer length";return false;}
    }
    for(uint32_t i=0;i<blocks-mtp;++i){
        const bool linear=(i+1)%interval!=0;
        if(rv&&bool((*rv)[i])!=linear){err="GGUF recurrent_layers contradicts full_attention_interval at layer "+std::to_string(i);return false;}
        layers.array.push_back(str(linear?"linear_attention":"full_attention"));
    }
    put(tc,"layer_types",std::move(layers));
    put(root,"model_type",str("qwen3_5_moe"));
    // The existing parser reads MoE scalar keys from the outer document.
    for(const char* k:{"num_experts","num_experts_per_tok","moe_intermediate_size",
                        "shared_expert_intermediate_size","tie_word_embeddings"})
        put(root,k,*tc.find(k));
    put(root,"text_config",tc);
    if(!sidecar.empty()) {
        try {
            Json hf=Json::parse(sidecar);
            const Json* model_type=hf.find("model_type");
            if(!model_type||!equal_json(*root.find("model_type"),*model_type)){
                err="GGUF/config.json mismatch model_type: GGUF="+root.at("model_type").dump()+
                    " config.json="+(model_type?model_type->dump():"<missing>");return false;
            }
            const Json* ht=hf.find("text_config");if(!ht)ht=&hf;
            for(const auto& kv:tc.object) {
                const Json* actual=ht->find(kv.first);
                if(kv.first=="rope_parameters") {
                    if(!actual){err="config.json missing rope_parameters";return false;}
                    for(const auto& r:kv.second.object) {
                        const Json* hv=actual->find(r.first);
                        if(!hv||!equal_json(r.second,*hv)) {
                            err="GGUF/config.json mismatch "+r.first+": GGUF="+r.second.dump()+
                                " config.json="+(hv?hv->dump():"<missing>");return false;
                        }
                    }
                } else if(!actual && kv.first=="rope_theta") {
                    // HF keeps this under rope_parameters in this release.
                    continue;
                } else if(!actual||!equal_json(kv.second,*actual)) {
                    err="GGUF/config.json mismatch "+kv.first+": GGUF="+kv.second.dump()+
                        " config.json="+(actual?actual->dump():"<missing>");return false;
                }
            }
        } catch(const std::exception& e){err=std::string("config.json parse: ")+e.what();return false;}
    }
    out=root.dump();return true;
}

bool gguf_qwen35_index(Qwen35Model& m,std::string& err) {
    using I=TensorRef::GGUFInverse;
    if(!m.gguf_file){err="GGUF source missing";return false;}
    auto add=[&](const GGUFTensor& t,const std::string& name,I inv=I::NONE,int expert=-1){
        TensorRef r;r.gguf=&t;r.gguf_inverse=inv;r.gguf_expert=expert;
        r.t.name=name;r.t.dtype=STDtype::UNKNOWN;
        if(expert>=0)r.t.shape={int64_t(t.ne[1]),int64_t(t.ne[0])};
        else if(t.n_dims==1)r.t.shape={int64_t(t.ne[0])};
        else r.t.shape={int64_t(t.ne[1]),int64_t(t.ne[0])};
        if(name.find("shared_expert_gate.weight")!=std::string::npos && t.n_dims==1)
            r.t.shape={1,int64_t(t.ne[0])};
        r.t.begin=0;r.t.end=uint64_t(r.t.numel())*sizeof(float);
        if(expert>=0){
            if(name.find(".down_proj.")!=std::string::npos)
                r.t.shape={m.cfg.hidden,m.cfg.moe_inter};
            else r.t.shape={m.cfg.moe_inter,m.cfg.hidden};
            r.t.end=uint64_t(r.t.numel())*sizeof(float);
        }
        m.index.emplace(name,std::move(r));
    };
    const std::string p="model.language_model.";
    for(const auto& t:m.gguf_file->tensors()) {
        const std::string& s=t.name;
        if(s.rfind("nextn.",0)==0||
           s.rfind("blk."+std::to_string(m.cfg.n_layers)+".",0)==0)continue;
        if(t.type!=0&&t.type!=1&&t.type!=8&&t.type!=12&&t.type!=13&&
           t.type!=14&&t.type!=30){
            err="GGUF tensor "+s+" uses unsupported decode type "+
                (gguf_type_info(t.type)?std::string(gguf_type_info(t.type)->name):std::to_string(t.type));
            return false;
        }
        if(s=="token_embd.weight"){add(t,p+"embed_tokens.weight");continue;}
        if(s=="output.weight"){add(t,"lm_head.weight");continue;}
        if(s=="output_norm.weight"){add(t,p+"norm.weight",I::NORM);continue;}
        if(s.rfind("blk.",0)!=0){err="unmapped GGUF tensor: "+s;return false;}
        const size_t dot=s.find('.',4);
        if(dot==std::string::npos){err="unmapped GGUF tensor: "+s;return false;}
        int l=-1;try{l=std::stoi(s.substr(4,dot-4));}catch(...){err="unmapped GGUF tensor: "+s;return false;}
        if(l>=m.cfg.n_layers && l<m.cfg.n_layers+1)continue; // MTP block
        if(l<0||l>=m.cfg.n_layers){err="unmapped GGUF layer tensor: "+s;return false;}
        const std::string role=s.substr(dot+1);
        const std::string b=p+"layers."+std::to_string(l)+".";
        auto one=[&](const char* from,const char* to,I inv=I::NONE){
            if(role==from){add(t,b+to,inv);return true;}return false;};
        if(one("attn_norm.weight","input_layernorm.weight",I::NORM)||
           one("post_attention_norm.weight","post_attention_layernorm.weight",I::NORM)||
           one("attn_qkv.weight","linear_attn.in_proj_qkv.weight",I::QKV_ROWS)||
           one("attn_gate.weight","linear_attn.in_proj_z.weight",I::V_ROWS)||
           one("ssm_alpha.weight","linear_attn.in_proj_a.weight",I::V_ROWS)||
           one("ssm_beta.weight","linear_attn.in_proj_b.weight",I::V_ROWS)||
           one("ssm_conv1d.weight","linear_attn.conv1d.weight",I::CONV_ROWS)||
           one("ssm_a","linear_attn.A_log",I::A_LOG)||
           one("ssm_dt.bias","linear_attn.dt_bias",I::V_VECTOR)||
           one("ssm_norm.weight","linear_attn.norm.weight")||
           one("ssm_out.weight","linear_attn.out_proj.weight",I::V_COLS)||
           one("attn_q.weight","self_attn.q_proj.weight")||
           one("attn_k.weight","self_attn.k_proj.weight")||
           one("attn_v.weight","self_attn.v_proj.weight")||
           one("attn_output.weight","self_attn.o_proj.weight")||
           one("attn_q_norm.weight","self_attn.q_norm.weight",I::NORM)||
           one("attn_k_norm.weight","self_attn.k_norm.weight",I::NORM)||
           one("ffn_gate_inp.weight","mlp.gate.weight")||
           one("ffn_gate_inp_shexp.weight","mlp.shared_expert_gate.weight")||
           one("ffn_gate_shexp.weight","mlp.shared_expert.gate_proj.weight")||
           one("ffn_up_shexp.weight","mlp.shared_expert.up_proj.weight")||
           one("ffn_down_shexp.weight","mlp.shared_expert.down_proj.weight"))continue;
        const std::string base=b+"mlp.experts.";
        auto experts=[&](const char* from,const char* to){
            if(role!=from)return false;
            const bool down=role=="ffn_down_exps.weight";
            if(t.n_dims!=3||t.ne[2]!=uint64_t(m.cfg.n_experts)||
               t.ne[0]!=uint64_t(down?m.cfg.moe_inter:m.cfg.hidden)||
               t.ne[1]!=uint64_t(down?m.cfg.hidden:m.cfg.moe_inter)){
                err="GGUF expert shape mismatch: "+s;return true;
            }
            for(int e=0;e<m.cfg.n_experts;++e){
                const std::string name=base+std::to_string(e)+"."+to+".weight";
                add(t,name,I::NONE,e);
                m.index[name].gguf_half=role=="ffn_gate_exps.weight"?0:
                                          role=="ffn_up_exps.weight"?1:-1;
            }
            return true;
        };
        if(experts("ffn_gate_exps.weight","gate_proj")||
           experts("ffn_up_exps.weight","up_proj")||
           experts("ffn_down_exps.weight","down_proj")){
            if(!err.empty())return false;continue;
        }
        err="unmapped GGUF tensor: "+s;return false;
    }
    return true;
}

bool Qwen35Model::read_gguf_raw_rows(const TensorRef& r,uint8_t* dst,std::string& err,
                                     uint64_t first,uint64_t count) const {
    if(!gguf_file||!r.gguf||!dst){err="GGUF tensor source missing";return false;}
    const auto& t=*r.gguf;
    const uint64_t K=uint64_t(r.t.shape.back()),N=uint64_t(r.t.numel())/K;
    if(!count)count=N-first;
    if(first>N||count>N-first||t.ne[0]!=K){err="GGUF raw row shape mismatch: "+t.name;return false;}
    if(K%256 || (t.type!=12 && t.type!=14)){
        err="native GGUF requires Q4_K/Q6_K and K divisible by 256: "+t.name;return false;
    }
    const int nk=cfg.lin_k_heads,nv=cfg.lin_v_heads,dv=cfg.lin_v_dim,dk=cfg.lin_k_dim;
    if(nk<=0||nv<=0||nv%nk){err="GGUF V-head geometry invalid";return false;}
    const auto inv=r.gguf_inverse;
    for(uint64_t i=0;i<count;++i){
        uint64_t row=first+i;
        if(inv==TensorRef::GGUFInverse::V_ROWS || inv==TensorRef::GGUFInverse::QKV_ROWS){
            const uint64_t start=inv==TensorRef::GGUFInverse::V_ROWS?0:uint64_t(2)*nk*dk;
            const uint64_t width=(t.name.find("ssm_alpha")!=std::string::npos||
                                  t.name.find("ssm_beta")!=std::string::npos)?1:dv;
            if(row>=start)row=start+gguf_v_tiled_index((row-start)/width,nk,nv)*width+
                                  (row-start)%width;
        }
        if(r.gguf_expert>=0)row+=uint64_t(r.gguf_expert)*N;
        if(row>=t.n_rows){err="GGUF raw row permutation out of bounds: "+t.name;return false;}
        if(!gguf_file->read_raw_row(t,row,dst+i*t.row_bytes,err))return false;
    }
    return true;
}

bool Qwen35Model::read_gguf_f32(const TensorRef& r,float* dst,std::string& err,
                                uint64_t first,uint64_t count) const {
    if(!gguf_file||!r.gguf||!dst){err="GGUF tensor source missing";return false;}
    const auto& t=*r.gguf;
    const uint64_t K=uint64_t(r.t.shape.back()),N=uint64_t(r.t.numel())/K;
    if(!count)count=N-first;
    if(first>N||count>N-first){err="GGUF logical row range out of bounds";return false;}
    if(t.ne[0]!=K && !(t.n_dims==1&&t.ne[0]==K)){
        err="GGUF row width mismatch: "+t.name;return false;
    }
    const int nk=cfg.lin_k_heads,nv=cfg.lin_v_heads,dv=cfg.lin_v_dim,dk=cfg.lin_k_dim;
    if(nk<=0||nv<=0||nv%nk){err="GGUF V-head geometry invalid";return false;}
    const uint64_t qk=uint64_t(2)*nk*dk;
    std::vector<uint8_t> raw(size_t(t.row_bytes));
    std::vector<float> decoded(size_t(K),0.0f);
    const auto inv=r.gguf_inverse;
    auto source_row=[&](uint64_t row){
        if(inv==TensorRef::GGUFInverse::V_ROWS || inv==TensorRef::GGUFInverse::QKV_ROWS ||
           inv==TensorRef::GGUFInverse::CONV_ROWS){
            const uint64_t start=inv==TensorRef::GGUFInverse::V_ROWS?0:qk;
            const uint64_t width=(t.name.find("ssm_alpha")!=std::string::npos||
                                  t.name.find("ssm_beta")!=std::string::npos)?1:dv;
            if(row>=start)return start+gguf_v_tiled_index((row-start)/width,nk,nv)*width+
                                  (row-start)%width;
        }
        return row;
    };
    for(uint64_t i=0;i<count;++i) {
        const uint64_t logical=first+i;
        uint64_t physical=source_row(logical);
        if(r.gguf_expert>=0)physical+=uint64_t(r.gguf_expert)*N;
        if(physical>=t.n_rows){err="GGUF row permutation out of bounds: "+t.name;return false;}
        if(!gguf_file->read_raw_row(t,physical,raw.data(),err)||
           !gguf_dequantize(t.type,raw.data(),K,decoded.data(),err,t.name))return false;
        float* out=dst+i*K;
        if(inv==TensorRef::GGUFInverse::V_COLS || inv==TensorRef::GGUFInverse::V_VECTOR ||
           inv==TensorRef::GGUFInverse::A_LOG) {
            const uint64_t width=inv==TensorRef::GGUFInverse::V_COLS?uint64_t(dv):1;
            if(K!=uint64_t(nv)*width){err="GGUF V column width mismatch: "+t.name;return false;}
            for(uint64_t k=0;k<K;++k)
                out[k]=decoded[gguf_v_tiled_index(k/width,nk,nv)*width+k%width];
        } else std::memcpy(out,decoded.data(),size_t(K)*sizeof(float));
        if(inv==TensorRef::GGUFInverse::NORM || inv==TensorRef::GGUFInverse::A_LOG){
            uint64_t bad=0;
            for(uint64_t k=0;k<K;++k){
                const float g=out[k];
                if(inv==TensorRef::GGUFInverse::NORM){
                    const float w=bf16_to_f32(f32_to_bf16(g-1.0f));
                    out[k]=w;float back=1.0f+w;
                    if(f32_to_bits(back)!=f32_to_bits(g))++bad;
                } else {
                    if(!(g<0.0f)){err="GGUF ssm_a must be negative";return false;}
                    const float w=bf16_to_f32(f32_to_bf16(float(std::log(-double(g)))));
                    out[k]=w;
                    if(!gguf_a_log_recovery_ok(g,w))++bad;
                }
            }
            if(inv==TensorRef::GGUFInverse::NORM){gguf_norm_checks+=K;gguf_norm_mismatches+=bad;}
            else {gguf_a_checks+=K;gguf_a_mismatches+=bad;}
            if(bad)std::printf("  GGUF %s round-trip mismatches: %llu/%llu\n",t.name.c_str(),
                               (unsigned long long)bad,(unsigned long long)K);
        }
    }
    return true;
}
}
