// GRIMOIRE -- qwen35/qwen35moe GGUF source adapter. SPDX-License-Identifier: GPL-3.0-or-later
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
    if(arch!="qwen35moe"&&arch!="qwen35") {err="unsupported GGUF architecture: "+arch;return false;}
    const bool moe=arch=="qwen35moe";
    const std::string p=arch+".";
    auto get=[&](const char* key,uint32_t& v){return f.get_u32(p+key,v,err);};
    uint32_t blocks,mtp=0,h,heads,kv,hd,vd,rope,ne=0,top=0,inter=0,shared=0,
             dense_inter=0,nk,nv,dk,inner,conv,interval;
    if(!get("block_count",blocks)||
       ((moe||f.get(p+"nextn_predict_layers"))&&!get("nextn_predict_layers",mtp))||
       mtp>=blocks||
       !get("embedding_length",h)||!get("attention.head_count",heads)||
       !get("attention.head_count_kv",kv)||!get("attention.key_length",hd)||
       !get("attention.value_length",vd)||!get("rope.dimension_count",rope)||
       !(moe ? (get("expert_count",ne)&&get("expert_used_count",top)&&
                 get("expert_feed_forward_length",inter)&&
                 get("expert_shared_feed_forward_length",shared))
              : get("feed_forward_length",dense_inter))||
       !get("ssm.group_count",nk)||!get("ssm.time_step_rank",nv)||
       !get("ssm.state_size",dk)||!get("ssm.inner_size",inner)||
       !get("ssm.conv_kernel",conv)||!get("full_attention_interval",interval))return false;
    float theta,eps;
    if(!f.get_f32(p+"rope.freq_base",theta,err)||
       !f.get_f32(p+"attention.layer_norm_rms_epsilon",eps,err)) return false;
    if(!hd||hd!=vd||!nk||!nv||nv%nk||inner%nv||!interval||!h||!heads||!kv||
       (moe ? (!ne||!top||!inter||!shared) : !dense_inter)) {
        err="inconsistent "+arch+" GGUF geometry";return false;
    }
    std::vector<uint32_t> sections;
    if(!array_u32(f,(p+"rope.dimension_sections").c_str(),sections,err))return false;
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
    if(moe){
        n("num_experts",ne);n("num_experts_per_tok",top);n("moe_intermediate_size",inter);
        n("shared_expert_intermediate_size",shared);
    } else n("intermediate_size",dense_inter);
    n("linear_num_key_heads",nk);
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
    put(root,"model_type",str(moe?"qwen3_5_moe":"qwen3_5"));
    if(!moe)put(tc,"model_type",str("qwen3_5_text"));
    // The existing parser reads MoE scalar keys and tying from the outer document.
    if(moe)for(const char* k:{"num_experts","num_experts_per_tok","moe_intermediate_size",
                            "shared_expert_intermediate_size"})put(root,k,*tc.find(k));
    put(root,"tie_word_embeddings",*tc.find("tie_word_embeddings"));
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
                } else if(kv.first=="mtp_num_hidden_layers" && actual &&
                          kv.second.kind==Json::Kind::Number && kv.second.number==0) {
                    // A GGUF may ship WITHOUT the MTP head the original model has
                    // (unsloth/Qwen3.8-27B-GGUF UD-IQ1_S / UD-IQ2_XXS: block_count 64,
                    // nextn_predict_layers 0, no nextn.* tensors).  The GGUF is what
                    // gets loaded, so 0 is authoritative; only "GGUF has MORE MTP
                    // layers than the model" would be a contradiction.
                    std::printf("  GGUF ships no MTP head (config.json has %s); MTP unavailable for this file\n",
                                actual->dump().c_str());
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

// Standard Qwen3 uses full-head NeoX RoPE and plain RMSNorm weights. Its
// converter inherits Qwen2/Qwen2Moe: no Q/K row permutation or norm shift.
bool gguf_qwen3_config(const GGUFFile& f,const std::string& sidecar,
                       std::string& out,std::string& err) {
    std::string arch;
    if(!f.get_string("general.architecture",arch,err))return false;
    if(arch!="qwen3"&&arch!="qwen3moe"){
        err="unsupported GGUF architecture: "+arch;return false;
    }
    const bool moe=arch=="qwen3moe";
    const std::string p=arch+".";
    auto u=[&](const char* k,uint32_t& v){return f.get_u32(p+k,v,err);};
    uint32_t blocks=0,h=0,ff=0,heads=0,kv=0,hd=0,vd=0,ctx=0,
             ne=0,top=0,ei=0,shared=0;
    float theta=0,eps=0;
    if(!u("block_count",blocks)||!u("embedding_length",h)||
       !u("feed_forward_length",ff)||!u("attention.head_count",heads)||
       !u("attention.head_count_kv",kv)||!u("attention.key_length",hd)||
       !u("attention.value_length",vd)||!u("context_length",ctx)||
       !f.get_f32(p+"rope.freq_base",theta,err)||
       !f.get_f32(p+"attention.layer_norm_rms_epsilon",eps,err)||
       (moe && (!u("expert_count",ne)||!u("expert_used_count",top)||
                !u("expert_feed_forward_length",ei))))return false;
    // Standard Qwen3-MoE has no shared expert. Older GGUFs may omit
    // the zero-valued metadata key altogether.
    if(moe && f.get(p+"expert_shared_feed_forward_length") &&
       !u("expert_shared_feed_forward_length",shared))return false;
    if(!blocks||!h||!ff||!heads||!kv||heads%kv||!hd||hd!=vd||
       !ctx||theta<=0||eps<=0||(moe&&(!ne||!top||top>ne||!ei))){
        err="inconsistent "+arch+" GGUF geometry";return false;
    }
    if(shared){err=arch+" expert_shared_feed_forward_length is nonzero; "+
                    "standard Qwen3-MoE has no shared expert";return false;}
    const GGUFTensor* emb=f.find("token_embd.weight");
    if(!emb||emb->n_dims!=2||emb->ne[0]!=h){
        err=arch+" token_embd.weight shape disagrees with embedding_length";return false;
    }
    const bool tied=!f.find("output.weight");
    std::vector<uint32_t> dense_layers;
    for(uint32_t i=0;i<blocks;++i){
        const auto* q=f.find("blk."+std::to_string(i)+".attn_q.weight");
        if(!q||q->n_dims!=2||q->ne[0]!=h||q->ne[1]!=uint64_t(heads)*hd){
            err=arch+" attn_q.weight must have heads*head_dim rows (no output gate), layer "+
                std::to_string(i);return false;
        }
        if(moe){
            const std::string b="blk."+std::to_string(i)+".";
            if(!f.find(b+"ffn_gate_inp.weight")){
                for(const char* role:{"ffn_gate.weight","ffn_up.weight","ffn_down.weight"})
                    if(!f.find(b+role)){
                        err=arch+" layer "+std::to_string(i)+" has neither a router nor "+role;
                        return false;
                    }
                dense_layers.push_back(i);
            }
        }
    }
    Json root=obj();
    auto n=[&](const char* k,double v){put(root,k,num(v));};
    put(root,"model_type",str(moe?"qwen3_moe":"qwen3"));
    n("num_hidden_layers",blocks);n("hidden_size",h);n("intermediate_size",ff);
    n("vocab_size",emb->ne[1]);n("num_attention_heads",heads);
    n("num_key_value_heads",kv);n("head_dim",hd);n("rope_theta",theta);
    n("partial_rotary_factor",1);n("rms_norm_eps",eps);
    n("max_position_embeddings",ctx);
    put(root,"hidden_act",str("silu"));
    put(root,"attention_bias",flag(false));
    put(root,"attn_output_gate",flag(false));
    put(root,"tie_word_embeddings",flag(tied));
    if(moe){
        n("num_experts",ne);n("num_experts_per_tok",top);
        n("moe_intermediate_size",ei);n("shared_expert_intermediate_size",0);
        n("decoder_sparse_step",1);
        Json mlp_only=arr();
        for(uint32_t i:dense_layers)mlp_only.array.push_back(num(i));
        put(root,"norm_topk_prob",flag(true));put(root,"mlp_only_layers",mlp_only);
    }
    if(!sidecar.empty()){
        try{
            Json hf=Json::parse(sidecar);
            if(moe){
                const Json* step=hf.find("decoder_sparse_step");
                const Json* only=hf.find("mlp_only_layers");
                const int sparse_step=step?step->integer():1;
                if(sparse_step<=0){err="config.json decoder_sparse_step must be positive";return false;}
                for(uint32_t i=0;i<blocks;++i){
                    bool mlp_only=false;
                    if(only && only->kind==Json::Kind::Array)
                        for(const Json& x:only->array)mlp_only|=x.integer()==int(i);
                    const bool hf_dense=mlp_only||(i+1)%sparse_step!=0;
                    const bool gguf_dense=std::find(dense_layers.begin(),dense_layers.end(),i)!=dense_layers.end();
                    if(hf_dense!=gguf_dense){
                        err="GGUF/config.json MoE layer pattern differs at layer "+std::to_string(i);
                        return false;
                    }
                }
            }
            for(const auto& kvp:root.object){
                // The Coder GGUF records feed_forward_length=5472 while
                // HF config says intermediate_size=6144. All 48 layers are
                // routed, so neither dense width is used by this file.
                if(moe&&kvp.first=="intermediate_size"&&dense_layers.empty())continue;
                if(moe&&(kvp.first=="decoder_sparse_step"||kvp.first=="mlp_only_layers"))continue;
                const Json* actual=hf.find(kvp.first);
                if(!actual || !equal_json(kvp.second,*actual)){
                    // These two values are converter defaults, absent in HF.
                    if((kvp.first=="partial_rotary_factor"||
                        kvp.first=="attn_output_gate"||
                        kvp.first=="shared_expert_intermediate_size")&&!actual)continue;
                    err="GGUF/config.json mismatch "+kvp.first+": GGUF="+kvp.second.dump()+
                        " config.json="+(actual?actual->dump():"<missing>");return false;
                }
            }
        }catch(const std::exception& e){err=std::string("config.json parse: ")+e.what();return false;}
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
    const std::string p=m.cfg.is_qwen3 ? "model." : "model.language_model.";
    const I norm_inv=m.cfg.is_qwen3 ? I::NONE : I::NORM;
    for(const auto& t:m.gguf_file->tensors()) {
        const std::string& s=t.name;
        if(s.rfind("nextn.",0)==0)continue;
        // The MTP block (blk.<n_layers>.*, nextn_predict_layers 1) is mapped below when the
        // config has it; without mtp_num_hidden_layers it stays unused, as before.
        if(s.rfind("blk."+std::to_string(m.cfg.n_layers)+".",0)==0&&m.cfg.mtp_n_layers<1)continue;
        // Ask the decoder itself whether it can decode this type (one zero
        // block), instead of keeping a second list that drifts from it: a
        // hard-coded 7-type list here refused Unsloth's Q4_0/IQ* files after
        // gguf_decode had grown to every ggml type.
        auto decodable=[](uint32_t type,const std::string& name){
            const auto* ti=gguf_type_info(type);
            if(!ti)return false;
            std::vector<uint8_t> blk(ti->block_bytes,0);
            std::vector<float> out(ti->block_elems);
            std::string e;
            return gguf_dequantize(type,blk.data(),ti->block_elems,out.data(),e,name);
        };
        if(!decodable(t.type,s)){
            err="GGUF tensor "+s+" uses unsupported decode type "+
                (gguf_type_info(t.type)?std::string(gguf_type_info(t.type)->name):std::to_string(t.type));
            return false;
        }
        if(s=="token_embd.weight"){add(t,p+"embed_tokens.weight");continue;}
        if(s=="output.weight"){add(t,"lm_head.weight");continue;}
        if(s=="output_norm.weight"){add(t,p+"norm.weight",norm_inv);continue;}
        if(s.rfind("blk.",0)!=0){err="unmapped GGUF tensor: "+s;return false;}
        const size_t dot=s.find('.',4);
        if(dot==std::string::npos){err="unmapped GGUF tensor: "+s;return false;}
        int l=-1;try{l=std::stoi(s.substr(4,dot-4));}catch(...){err="unmapped GGUF tensor: "+s;return false;}
        const bool mtp_block=l==m.cfg.n_layers&&m.cfg.mtp_n_layers>=1;
        if(l<0||(l>=m.cfg.n_layers&&!mtp_block)){err="unmapped GGUF layer tensor: "+s;return false;}
        const std::string role=s.substr(dot+1);
        // MTP block (2026-10-29): llama.cpp's names for the HF mtp.* tensors.  The layer's own
        // tensors take the transformer layers' roles under mtp.layers.0.; nextn.eh_proj is
        // mtp.fc ([H][2H], same order); the three nextn norms carry the +1 of every zero-centred
        // Qwen3.5 norm (norm_inv) like the layers' norms.  The head shares the main embedding
        // and output head (a nextn copy of either, if a file has one, is not needed).
        if(mtp_block&&role.rfind("nextn.",0)==0){
            if(role=="nextn.eh_proj.weight"){add(t,"mtp.fc.weight");continue;}
            if(role=="nextn.enorm.weight"){add(t,"mtp.pre_fc_norm_embedding.weight",norm_inv);continue;}
            if(role=="nextn.hnorm.weight"){add(t,"mtp.pre_fc_norm_hidden.weight",norm_inv);continue;}
            if(role=="nextn.shared_head_norm.weight"){add(t,"mtp.norm.weight",norm_inv);continue;}
            if(role=="nextn.embed_tokens.weight"||role=="nextn.shared_head_head.weight")continue;
            err="unmapped GGUF MTP tensor: "+s;return false;
        }
        const std::string b=mtp_block ? std::string("mtp.layers.0.") : p+"layers."+std::to_string(l)+".";
        const bool dense_layer=!m.cfg.is_moe() ||
            (m.cfg.is_qwen3 && std::find(m.cfg.mlp_only_layers.begin(),
                                        m.cfg.mlp_only_layers.end(),l)!=m.cfg.mlp_only_layers.end());
        if(dense_layer && (role=="ffn_gate.weight"||role=="ffn_up.weight"||
                           role=="ffn_down.weight")){
            const bool down=role=="ffn_down.weight";
            if(t.n_dims!=2||t.ne[0]!=uint64_t(down?m.cfg.dense_inter:m.cfg.hidden)||
               t.ne[1]!=uint64_t(down?m.cfg.hidden:m.cfg.dense_inter)){
                err="GGUF dense FFN shape mismatch: "+s;return false;
            }
        }
        auto one=[&](const char* from,const char* to,I inv=I::NONE){
            if(role==from){add(t,b+to,inv);return true;}return false;};
        if(one("attn_norm.weight","input_layernorm.weight",norm_inv)||
           one("post_attention_norm.weight","post_attention_layernorm.weight",norm_inv)||
           (m.cfg.is_qwen3&&one("ffn_norm.weight","post_attention_layernorm.weight"))||
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
           one("attn_q_norm.weight","self_attn.q_norm.weight",norm_inv)||
           one("attn_k_norm.weight","self_attn.k_norm.weight",norm_inv)||
           (dense_layer && (one("ffn_gate.weight","mlp.gate_proj.weight")||
                                one("ffn_up.weight","mlp.up_proj.weight")||
                                one("ffn_down.weight","mlp.down_proj.weight")))||
           (m.cfg.is_moe() && !dense_layer &&
            (one("ffn_gate_inp.weight","mlp.gate.weight")||
             one("ffn_gate_inp_shexp.weight","mlp.shared_expert_gate.weight")||
             one("ffn_gate_shexp.weight","mlp.shared_expert.gate_proj.weight")||
             one("ffn_up_shexp.weight","mlp.shared_expert.up_proj.weight")||
             one("ffn_down_shexp.weight","mlp.shared_expert.down_proj.weight"))))continue;
        if(dense_layer){err="unmapped GGUF tensor: "+s;return false;}
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

bool gguf_qwen3_index(Qwen35Model& m,std::string& err){
    return gguf_qwen35_index(m,err);
}

bool Qwen35Model::read_gguf_raw_rows(const TensorRef& r,uint8_t* dst,std::string& err,
                                     uint64_t first,uint64_t count) const {
    if(!gguf_file||!r.gguf||!dst){err="GGUF tensor source missing";return false;}
    const auto& t=*r.gguf;
    const uint64_t K=uint64_t(r.t.shape.back()),N=uint64_t(r.t.numel())/K;
    if(!count)count=N-first;
    if(first>N||count>N-first||t.ne[0]!=K){err="GGUF raw row shape mismatch: "+t.name;return false;}
    if(!is_native_gguf_type(t.type) || K % native_gguf_block_elems(native_gguf_fmt(t.type))){
        err="native GGUF requires supported block format and aligned K: "+t.name;return false;
    }
    const int nk=cfg.lin_k_heads,nv=cfg.lin_v_heads,dv=cfg.lin_v_dim,dk=cfg.lin_k_dim;
    const auto inv=r.gguf_inverse;
    if(inv!=TensorRef::GGUFInverse::NONE &&
       (nk<=0||nv<=0||nv%nk)){err="GGUF V-head geometry invalid";return false;}
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
    const auto inv=r.gguf_inverse;
    if(inv!=TensorRef::GGUFInverse::NONE &&
       (nk<=0||nv<=0||nv%nk)){err="GGUF V-head geometry invalid";return false;}
    const uint64_t qk=uint64_t(2)*nk*dk;
    std::vector<uint8_t> raw(size_t(t.row_bytes));
    std::vector<float> decoded(size_t(K),0.0f);
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

// Gemma 4's GGUF has the same row-major projection payloads as HF. The
// converter renames tensors but (unlike Gemma 3) applies no norm offset.
bool gguf_gemma4_config(const GGUFFile& f,const std::string& sidecar,
                        std::string& out,std::string& err) {
    std::string arch;
    if(!f.get_string("general.architecture",arch,err))return false;
    if(arch!="gemma4"){err="unsupported GGUF architecture: "+arch;return false;}
    const std::string p="gemma4.";
    auto u=[&](const char* k,uint32_t& v){return f.get_u32(p+k,v,err);};
    auto fl=[&](const char* k,float& v){return f.get_f32(p+k,v,err);};
    uint32_t blocks=0,h=0,inter=0,heads=0,full_hd=0,swa_hd=0,full_v=0,swa_v=0,
             rot_full=0,rot_swa=0,window=0,shared=0,per_input=0,ctx=0;
    float theta_full=0,theta_swa=0,eps=0,softcap=0;
    if(!u("block_count",blocks)||!u("embedding_length",h)||
       !u("feed_forward_length",inter)||!u("attention.head_count",heads)||
       !u("attention.key_length",full_hd)||!u("attention.key_length_swa",swa_hd)||
       !u("attention.value_length",full_v)||!u("attention.value_length_swa",swa_v)||
       !u("rope.dimension_count",rot_full)||!u("rope.dimension_count_swa",rot_swa)||
       !u("attention.sliding_window",window)||!u("attention.shared_kv_layers",shared)||
       !u("embedding_length_per_layer_input",per_input)||!u("context_length",ctx)||
       !fl("rope.freq_base",theta_full)||!fl("rope.freq_base_swa",theta_swa)||
       !fl("attention.layer_norm_rms_epsilon",eps)||
       !fl("final_logit_softcapping",softcap))return false;
    if(shared){err="gemma4 attention.shared_kv_layers > 0 is unsupported";return false;}
    if(per_input){err="gemma4 embedding_length_per_layer_input > 0 is unsupported";return false;}
    if(f.get(p+"expert_count")||f.get(p+"expert_feed_forward_length")){
        err="gemma4 expert_count / MoE is unsupported";return false;
    }
    if(!blocks||!h||!inter||!heads||!swa_hd||!full_hd||
       full_v!=full_hd||swa_v!=swa_hd||rot_full!=full_hd||rot_swa!=swa_hd||
       !window||!ctx||theta_full<=0||theta_swa<=0||eps<=0||softcap<=0){
        err="inconsistent gemma4 attention, RoPE, or FFN geometry";return false;
    }
    const GGUFValue *patternv=nullptr;
    std::vector<uint32_t> kv;
    if(!array_u32(f,(p+"attention.head_count_kv").c_str(),kv,err)||
       !f.get_array(p+"attention.sliding_window_pattern",patternv,err))return false;
    const auto* pattern=std::get_if<std::vector<uint8_t>>(&patternv->data);
    if(!pattern||kv.size()!=blocks||pattern->size()!=blocks){
        err="gemma4 per-layer head_count_kv / sliding_window_pattern must have block_count entries";
        return false;
    }
    uint32_t full_kv=0,swa_kv=0;
    Json layers=arr();
    bool any_full=false,any_swa=false,full_v_present=false,full_v_absent=false;
    for(uint32_t i=0;i<blocks;++i){
        if((*pattern)[i]>1){err="gemma4 sliding_window_pattern is not boolean";return false;}
        const bool swa=(*pattern)[i]!=0;
        uint32_t& expected=swa?swa_kv:full_kv;
        if(!expected)expected=kv[i];
        if(!expected||expected!=kv[i]){
            err="gemma4 per_layer_config head_count_kv differs within attention type at layer "+std::to_string(i);
            return false;
        }
        any_swa|=swa;any_full|=!swa;
        layers.array.push_back(str(swa?"sliding_attention":"full_attention"));
        const std::string b="blk."+std::to_string(i)+".";
        const bool has_v=f.find(b+"attn_v.weight")!=nullptr;
        if(swa&&!has_v){err="gemma4 sliding layer missing attn_v.weight: "+b;return false;}
        if(!swa){full_v_present|=has_v;full_v_absent|=!has_v;}
    }
    if(!any_full||!any_swa||(full_v_present&&full_v_absent)){
        err="gemma4 mixed attention or attention_k_eq_v tensor pattern is inconsistent";return false;
    }
    const GGUFTensor* emb=f.find("token_embd.weight");
    if(!emb||emb->n_dims!=2||emb->ne[0]!=h){
        err="gemma4 token_embd.weight shape disagrees with embedding_length";return false;
    }
    const GGUFTensor* rf=f.find("rope_freqs.weight");
    if(!rf||rf->n_dims!=1||rf->ne[0]!=full_hd/2||rf->type!=0){
        err="gemma4 rope_freqs.weight must be F32 [global_head_dim/2]";return false;
    }
    // conversion/gemma.py:752-762 writes [1]*rotated_pairs +
    // [1e30]*unrotated_pairs. Header slices have no payload: 0.25 is the
    // converter's Gemma 4 default, checked against the HF sidecar below.
    double partial=0.25;
    if(rf->offset<=f.file_size() && rf->nbytes<=f.file_size()-rf->offset){
        std::vector<float> freq(size_t(rf->ne[0]));
        if(!f.read_raw(*rf,freq.data(),err))return false;
        size_t active=0;
        while(active<freq.size()&&freq[active]==1.0f)++active;
        if(!active||active==freq.size()){
            err="gemma4 rope_freqs.weight has no proportional RoPE cutoff";return false;
        }
        for(size_t i=active;i<freq.size();++i)if(freq[i]!=1e30f){
            err="gemma4 rope_freqs.weight has unsupported frequency factors";return false;
        }
        partial=double(2*active)/full_hd;
    }
    Json root=obj(),tc=obj(),rp=obj(),full=obj(),swa=obj();
    auto n=[&](const char* k,double v){put(tc,k,num(v));};
    n("num_hidden_layers",blocks);n("hidden_size",h);n("intermediate_size",inter);
    n("vocab_size",emb->ne[1]);n("num_attention_heads",heads);
    n("num_key_value_heads",swa_kv);n("num_global_key_value_heads",full_kv);
    n("head_dim",swa_hd);n("global_head_dim",full_hd);
    n("sliding_window",window);n("max_position_embeddings",ctx);
    n("rms_norm_eps",eps);n("final_logit_softcapping",softcap);
    n("num_kv_shared_layers",shared);n("hidden_size_per_layer_input",per_input);
    put(tc,"attention_k_eq_v",flag(full_v_absent));
    put(tc,"attention_bias",flag(false));put(tc,"enable_moe_block",flag(false));
    put(tc,"use_double_wide_mlp",flag(false));
    put(tc,"use_bidirectional_attention",str("vision"));
    put(tc,"hidden_activation",str("gelu_pytorch_tanh"));
    put(tc,"tie_word_embeddings",flag(!f.find("output.weight")));
    put(tc,"layer_types",std::move(layers));
    put(full,"rope_type",str("proportional"));
    put(full,"rope_theta",num(theta_full));
    put(full,"partial_rotary_factor",num(partial));
    put(swa,"rope_type",str("default"));put(swa,"rope_theta",num(theta_swa));
    put(rp,"full_attention",std::move(full));
    put(rp,"sliding_attention",std::move(swa));
    put(tc,"rope_parameters",std::move(rp));
    put(tc,"model_type",str("gemma4_unified_text"));
    put(root,"model_type",str("gemma4_unified"));
    put(root,"tie_word_embeddings",*tc.find("tie_word_embeddings"));
    put(root,"text_config",tc);
    if(!sidecar.empty())try{
        Json hf=Json::parse(sidecar);
        const Json* ht=hf.find("text_config");if(!ht)ht=&hf;
        for(const auto& entry:tc.object){
            const Json* actual=ht->find(entry.first);
            if(!actual||!equal_json(entry.second,*actual)){
                err="GGUF/config.json mismatch "+entry.first+": GGUF="+entry.second.dump()+
                    " config.json="+(actual?actual->dump():"<missing>");return false;
            }
        }
        const Json* tied=hf.find("tie_word_embeddings");
        if(tied&&!equal_json(*root.find("tie_word_embeddings"),*tied)){
            err="GGUF/config.json mismatch tie_word_embeddings";return false;
        }
    }catch(const std::exception& e){err=std::string("config.json parse: ")+e.what();return false;}
    out=root.dump();return true;
}

bool gguf_gemma4_index(Qwen35Model& m,std::string& err){
    if(!m.gguf_file){err="GGUF source missing";return false;}
    const std::string p="model.language_model.";
    auto add=[&](const GGUFTensor& t,const std::string& name){
        TensorRef r;r.gguf=&t;r.t.name=name;r.t.dtype=STDtype::UNKNOWN;
        if(t.n_dims==1)r.t.shape={int64_t(t.ne[0])};
        else r.t.shape={int64_t(t.ne[1]),int64_t(t.ne[0])};
        r.t.begin=0;r.t.end=uint64_t(r.t.numel())*sizeof(float);
        m.index.emplace(name,std::move(r));
    };
    for(const auto& t:m.gguf_file->tensors()){
        const std::string& s=t.name;
        auto shape=[&](uint64_t k,uint64_t n=0){
            if((n ? (t.n_dims==2&&t.ne[0]==k&&t.ne[1]==n)
                  : (t.n_dims==1&&t.ne[0]==k)))return true;
            err="gemma4 GGUF tensor shape mismatch: "+s;return false;
        };
        const auto* ti=gguf_type_info(t.type);
        if(!ti){err="unsupported GGUF tensor type: "+s;return false;}
        std::vector<uint8_t> raw(ti->block_bytes);
        std::vector<float> decoded(ti->block_elems);
        std::string why;
        if(!gguf_dequantize(t.type,raw.data(),ti->block_elems,decoded.data(),why,s)){
            err="unsupported GGUF tensor decode type: "+s+": "+why;return false;
        }
        if(s=="token_embd.weight"){
            if(m.cfg.hidden>0&&!shape(m.cfg.hidden,m.cfg.vocab))return false;
            add(t,p+"embed_tokens.weight");continue;
        }
        if(s=="output.weight"){
            if(m.cfg.hidden>0&&!shape(m.cfg.hidden,m.cfg.vocab))return false;
            add(t,"lm_head.weight");continue;
        }
        if(s=="output_norm.weight"){
            if(m.cfg.hidden>0&&!shape(m.cfg.hidden))return false;
            add(t,p+"norm.weight");continue;
        }
        if(s=="rope_freqs.weight")continue; // consumed and validated in config
        if(s.rfind("blk.",0)!=0){err="unmapped GGUF tensor: "+s;return false;}
        const size_t dot=s.find('.',4);
        uint32_t l=UINT32_MAX;
        try{l=uint32_t(std::stoul(s.substr(4,dot-4)));}catch(...){}
        if(dot==std::string::npos||l>=uint32_t(m.cfg.n_layers)){
            err="unmapped GGUF layer tensor: "+s;return false;
        }
        const std::string role=s.substr(dot+1);
        const std::string b=p+"layers."+std::to_string(l)+".";
        if(m.cfg.hidden>0){
            const uint64_t h=m.cfg.hidden, hd=m.cfg.layer_head_dim(int(l));
            const uint64_t q=uint64_t(m.cfg.n_heads)*hd;
            const uint64_t kv=uint64_t(m.cfg.layer_kv_heads(int(l)))*hd;
            const uint64_t ff=m.cfg.dense_inter;
            bool valid=true;
            if(role=="layer_output_scale.weight")valid=shape(1);
            else if(role=="attn_q_norm.weight"||role=="attn_k_norm.weight")valid=shape(hd);
            else if(role=="attn_norm.weight"||role=="post_attention_norm.weight"||
                    role=="ffn_norm.weight"||role=="post_ffw_norm.weight")valid=shape(h);
            else if(role=="attn_q.weight")valid=shape(h,q);
            else if(role=="attn_k.weight"||role=="attn_v.weight")valid=shape(h,kv);
            else if(role=="attn_output.weight")valid=shape(q,h);
            else if(role=="ffn_gate.weight"||role=="ffn_up.weight")valid=shape(h,ff);
            else if(role=="ffn_down.weight")valid=shape(ff,h);
            if(!valid)return false;
        }
        auto one=[&](const char* from,const char* to){
            if(role!=from)return false;add(t,b+to);return true;
        };
        if(one("attn_norm.weight","input_layernorm.weight")||
           one("post_attention_norm.weight","post_attention_layernorm.weight")||
           one("ffn_norm.weight","pre_feedforward_layernorm.weight")||
           one("post_ffw_norm.weight","post_feedforward_layernorm.weight")||
           one("layer_output_scale.weight","layer_scalar")||
           one("attn_q.weight","self_attn.q_proj.weight")||
           one("attn_k.weight","self_attn.k_proj.weight")||
           one("attn_v.weight","self_attn.v_proj.weight")||
           one("attn_output.weight","self_attn.o_proj.weight")||
           one("attn_q_norm.weight","self_attn.q_norm.weight")||
           one("attn_k_norm.weight","self_attn.k_norm.weight")||
           one("ffn_gate.weight","mlp.gate_proj.weight")||
           one("ffn_up.weight","mlp.up_proj.weight")||
           one("ffn_down.weight","mlp.down_proj.weight"))continue;
        err="unmapped GGUF tensor: "+s;return false;
    }
    return true;
}
}
