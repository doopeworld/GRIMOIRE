// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/inkling2.hpp"
#include "b70/json.hpp"
#include "b70/nvfp4.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace b70::inkling2 {
namespace {
void expect(bool ok,const std::string& message) {
    if(!ok) throw std::runtime_error("Inkling2 Mini: "+message);
}
int dim(const Json& j,const char* name,int max) {
    const int v=j.at(name).integer();
    expect(v>0 && v<=max,std::string("invalid ")+name);
    return v;
}
float scalar(const Json& j,const char* name,bool allow_zero=false) {
    const auto& v=j.at(name);
    expect(v.kind==Json::Kind::Number && std::isfinite(v.number) && v.number<=std::numeric_limits<float>::max() &&
        (allow_zero ? v.number>=0 : float(v.number)>0),std::string("invalid ")+name);
    return float(v.number);
}
Attention attention(const Json& j,const char* h,const char* kv,const char* d) {
    Attention a{dim(j,h,128),dim(j,kv,128),dim(j,d,256)};
    expect(a.heads%a.kv_heads==0,"query heads must be a multiple of KV heads");
    return a;
}
float sigmoid(float x) { return x>=0 ? 1.f/(1.f+std::exp(-x)) : std::exp(x)/(1.f+std::exp(x)); }
const STTensor& tensor(const TensorMap& in,const std::string& name) {
    auto it=in.find(name);
    expect(it!=in.end(),"missing tensor "+name);
    return it->second;
}
uint64_t nbytes(STDtype type,const std::vector<int64_t>& shape) {
    uint64_t bytes=st_dtype_size(type);
    expect(bytes>0,"unknown tensor dtype");
    for(auto d:shape) {
        expect(d>0 && bytes<=std::numeric_limits<uint64_t>::max()/uint64_t(d),"invalid tensor shape");
        bytes*=uint64_t(d);
    }
    return bytes;
}
void validate_ranges(const TensorMap& in) {
    expect(!in.empty(),"empty tensor header");
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    ranges.reserve(in.size());
    for(const auto& [name,t]:in) {
        expect(t.end>t.begin && t.end-t.begin==nbytes(t.dtype,t.shape),"invalid tensor range for "+name);
        ranges.emplace_back(t.begin,t.end);
    }
    std::sort(ranges.begin(),ranges.end());
    uint64_t prev=0;
    for(auto [a,b]:ranges) { expect(a==prev,"overlapping or gapped tensor offsets"); prev=b; }
}
void shape(const TensorMap& in,const std::string& name,STDtype type,std::initializer_list<int64_t> dims) {
    const auto& t=tensor(in,name);
    expect(t.dtype==type && t.shape==std::vector<int64_t>(dims) && t.end>=t.begin && t.end-t.begin==nbytes(t.dtype,t.shape),
           "wrong dtype/shape/byte count for "+name);
}
PackedFormat projection(const Config& c,const TensorMap& in,const std::string& name,int n,int k) {
    auto type=tensor(in,name).dtype;
    if(type==STDtype::U8) {
        expect(k%16==0,"NVFP4 width not divisible by 16");
        shape(in,name,STDtype::U8,{c.experts,n,k/2});
        shape(in,name+"_scale",STDtype::F8_E4M3,{c.experts,n,k/16});
        shape(in,name+"_amax",STDtype::F32,{c.experts});
        shape(in,name+"_outer_scale_denominator",STDtype::F32,{1});
        return PackedFormat::NVFP4;
    }
    expect(type==STDtype::F8_E4M3 && k%32==0,"unsupported expert format for "+name);
    shape(in,name,STDtype::F8_E4M3,{c.experts,n,k});
    shape(in,name+"_scale",STDtype::U8,{c.experts,n,k/32});
    expect(!in.count(name+"_amax") && !in.count(name+"_outer_scale_denominator"),"unexpected MXFP8 outer scale "+name);
    return PackedFormat::MXFP8;
}
} // namespace

Config parse_config(std::string_view document) {
    const Json root=Json::parse(document);
    expect(root.at("model_type").text()=="inkling2_loop","expected inkling2_loop checkpoint");
    const auto& j=root.at("text_config");
    expect(j.at("model_type").text()=="inkling2_loop_text","unexpected text architecture");
    const auto& loop=j.at("loop_config");
    expect(!loop.at("is_varloop").flag(),"variable passes not supported");
    expect(j.int_or("n_shared_experts",0)==0 && j.bool_or("use_gate_bias",true) &&
           !j.bool_or("q_bias",false) && !j.bool_or("o_bias",false) &&
           j.bool_or("use_value_norm_no_gamma",true) && j.bool_or("attn_res_normalize_values",true) &&
           j.text_or("scoring_func","sigmoid")=="sigmoid","unsupported text math configuration");
    Config c;
    c.hidden=dim(j,"hidden_size",8192); c.layers=dim(j,"num_hidden_layers",128);
    c.vocab=dim(j,"vocab_size",2<<20); c.output_vocab=dim(j,"unpadded_vocab_size",c.vocab);
    c.eos=root.at("eos_token_id").integer(); expect(c.eos>=0 && c.eos<c.output_vocab,"EOS in padding");
    c.max_context=dim(j,"model_max_length",1<<22);
    c.inter=dim(j,"intermediate_size",32768); c.experts=dim(j,"n_routed_experts",1024);
    c.top_k=dim(j,"num_experts_per_tok",c.experts); c.rel_dim=dim(j,"d_rel",256);
    c.rel_extent=dim(j,"rel_extent",c.max_context); c.window=dim(j,"sliding_window_size",c.max_context);
    c.conv=dim(j,"sconv_kernel_size",16); c.residual_block=dim(j,"n_layers_per_attn_res_block",c.layers);
    c.loop_start=dim(loop,"core_start",c.layers); c.loop_end=dim(loop,"core_end",c.layers);
    c.gate_rank=dim(loop,"gate_rank",c.hidden); c.passes=dim(loop,"num_passes",8);
    expect(c.loop_end>c.loop_start && c.loop_end<c.layers && c.passes>=2 && c.conv>=2 &&
           c.loop_start%c.residual_block==0 && c.loop_end%c.residual_block==0 && c.cache_slots()<=256,
           "invalid loop/residual block geometry");
    c.log_floor=dim(j,"log_scaling_n_floor",c.max_context); c.log_alpha=scalar(j,"log_scaling_alpha",true);
    c.eps=scalar(j,"rms_norm_eps"); c.rel_clip=scalar(j,"rel_l2_clip_threshold"); c.qk_clip=scalar(j,"qk_l2_clip_threshold");
    c.route_scale=scalar(j,"route_scale"); c.gate_cap=scalar(j,"mlp_gate_softcap"); c.up_cap=scalar(j,"mlp_up_softcap");
    c.logit_cap=scalar(j,"final_logit_softcapping");
    c.full=attention(j,"num_attention_heads","num_key_value_heads","head_dim");
    c.local=attention(j,"swa_num_attention_heads","swa_num_key_value_heads","swa_head_dim");
    const auto& local=j.at("local_layer_ids");
    expect(local.kind==Json::Kind::Array,"local_layer_ids is not an array");
    c.sliding.assign(c.layers,false);
    for(auto& item:local.array) {
        const int i=item.integer();
        expect(i>=0 && i<c.layers && !c.sliding[i],"invalid or duplicate local layer ID");
        c.sliding[i]=true;
    }
    return c;
}

std::vector<Visit> Config::visits() const {
    std::vector<Visit> out;
    auto add=[&](int i,int slot,int pass){out.push_back({i,slot,pass,sliding[i],i%residual_block==0,pass>0});};
    for(int i=0;i<loop_start;++i) add(i,i,-1);
    for(int p=0;p<passes;++p) for(int i=loop_start;i<loop_end;++i)
        add(i,p==0?i:layers+(p-1)*(loop_end-loop_start)+(i-loop_start),p);
    for(int i=loop_end;i<layers;++i) add(i,i,-1);
    return out;
}

TensorMap read_tensor_header(std::string_view input) {
    const Json root=Json::parse(input);
    expect(root.kind==Json::Kind::Object,"invalid safetensors header");
    TensorMap out;
    auto offset=[](const Json& j) {
        expect(j.kind==Json::Kind::Number && j.number>=0 && j.number<=9007199254740991. && std::trunc(j.number)==j.number,
               "invalid tensor offset");
        return uint64_t(j.number);
    };
    for (auto& [name,v]:root.object) {
        if(name=="__metadata__") continue;
        STTensor t;
        t.name=name;
        const auto d=v.at("dtype").text();
        for(auto dt:{STDtype::BF16,STDtype::F32,STDtype::F8_E4M3,STDtype::U8}) if(st_dtype_name(dt)==d) t.dtype=dt;
        if(d=="F8_E4M3FN") t.dtype=STDtype::F8_E4M3;
        const auto& dims=v.at("shape"); const auto& offsets=v.at("data_offsets");
        expect(dims.kind==Json::Kind::Array && dims.array.size()<=4 && offsets.kind==Json::Kind::Array && offsets.array.size()==2,
               "invalid header for "+name);
        for(auto& dim:dims.array) t.shape.push_back(dim.integer());
        t.begin=offset(offsets.array[0]); t.end=offset(offsets.array[1]);
        out.emplace(name,std::move(t));
    }
    validate_ranges(out);
    return out;
}

std::vector<LayerLayout> validate_tensors(const Config& c,const TensorMap& in) {
    validate_ranges(in);
    const std::string root="model.llm.";
    const int h=c.hidden;
    shape(in,root+"embed.weight",STDtype::BF16,{c.vocab,h});
    shape(in,root+"unembed.weight",STDtype::BF16,{c.vocab,h});
    for(auto name:{"norm.weight","embed_norm.weight"}) shape(in,root+name,STDtype::BF16,{h});
    shape(in,root+"attn_res.output_query",STDtype::F32,{h});
    for(auto name:{"middle_cycle_cross_pass_queries","middle_cycle_cross_pass_kv_queries"})
        shape(in,root+name,STDtype::F32,{2*(c.loop_end-c.loop_start),h});
    std::vector<LayerLayout> layouts;
    for(int i=0;i<c.layers;++i) {
        const auto a=c.sliding[i]?c.local:c.full;
        const int q=a.heads*a.head_dim,k=a.kv_heads*a.head_dim;
        const auto s=root+"layers."+std::to_string(i)+".";
        auto bf=[&](const std::string& name,std::initializer_list<int64_t> dims){shape(in,s+name,STDtype::BF16,dims);};
        if(i) {
            const auto name=root+"attn_res.attn_queries."+std::to_string(i-1);
            if(c.sliding[i]) shape(in,name,STDtype::F32,{h});
            else shape(in,name,STDtype::F32,{3,h});
        }
        shape(in,root+"attn_res.mlp_queries."+std::to_string(i),STDtype::F32,{h});
        bf("attn.wq_du.weight",{q,h}); bf("attn.wk_dv.weight",{k,h}); bf("attn.wv_dv.weight",{k,h});
        bf("attn.wo_ud.weight",{h,q}); bf("attn.gate_attn.weight",{q,h}); bf("attn.attn_sink_proj.weight",{a.heads,h});
        bf("attn.wr_du.weight",{a.heads*c.rel_dim,h}); bf("attn.q_l2_clip.scale",{a.heads,a.head_dim});
        bf("attn.rel_l2_clip.scale",{a.heads,c.rel_dim}); bf("attn.rel_logits_proj.proj",{c.rel_dim,c.sliding[i]?c.window:c.rel_extent});
        for(std::string kv:{"k","v"}) {
            if(c.sliding[i]) bf("attn."+kv+"_sconv.weight",{k,1,c.conv});
            else {
                bf("attn."+kv+"_sconv.static_w",{c.conv,k});
                bf("attn."+kv+"_sconv.weight_proj",{a.kv_heads*c.conv,h});
            }
        }
        if(c.sliding[i]) bf("attn_norm.weight",{h});
        else for(auto name:{"q_input_norm.weight","k_input_norm.weight","v_input_norm.weight"}) bf(name,{h});
        bf("attn_sconv.weight",{h,1,c.conv}); bf("mlp_sconv.weight",{h,1,c.conv}); bf("mlp_norm.weight",{h});
        bf("mlp.gate.weight",{c.experts,h}); shape(in,s+"mlp.gate.bias",STDtype::F32,{c.experts});
        layouts.push_back({projection(c,in,s+"mlp.experts.w13_weight",2*c.inter,h),
                           projection(c,in,s+"mlp.experts.w2_weight",h,c.inter)});
        if(i>=c.loop_start && i<c.loop_end) for(std::string name:{"attn","k","v","mlp"}) {
            const auto p="controllers."+name+".";
            bf(p+"down_weight",{c.gate_rank,2*h});
            bf(p+"gain_weight",{h,c.gate_rank}); bf(p+"forget_weight",{h,c.gate_rank});
        }
    }
    return layouts;
}

void PackedRows::row(int e,int r,std::span<float> out) const {
    expect(data && scale && cols>0 && e>=0 && e<experts && r>=0 && r<rows && out.size()==size_t(cols),
           "invalid packed row view or coordinates");
    const size_t i=size_t(e)*rows+r;
    if(fmt==PackedFormat::NVFP4) {
        expect(cols%16==0 && amax && std::isfinite(denominator) && denominator>0 && std::isfinite(amax[e]) && amax[e]>=0,"invalid NVFP4 row geometry or scale");
        const float outer=amax[e]/denominator;
        expect(std::isfinite(outer),"NVFP4 outer scale overflows");
        nvfp4_dequant_row(data+i*(cols/2),scale+i*(cols/16),outer,cols,out.data(),true);
    } else {
        expect(fmt==PackedFormat::MXFP8 && cols%32==0,"invalid MXFP8 row geometry");
        auto* d=data+i*cols;
        auto* s=scale+i*(cols/32);
        for(int k=0;k<cols;++k) out[k]=e4m3_to_f32(d[k])*e8m0_to_f32(s[k/32]);
    }
}

std::vector<Route> route(std::span<const float> logits,std::span<const float> bias,int k,float scale) {
    expect(logits.size()==bias.size() && k>0 && size_t(k)<=logits.size(),"invalid expert routing dimensions");
    std::vector<float> scores(logits.size());
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(),ids.end(),0);
    for(size_t i=0;i<scores.size();++i) {
        expect(std::isfinite(logits[i]) && std::isfinite(bias[i]),"nonfinite expert router input");
        scores[i]=sigmoid(logits[i]);
    }
    // Deterministic lower-ID tie break for the oracle. Do not claim PyTorch tie parity.
    std::partial_sort(ids.begin(),ids.begin()+k,ids.end(),[&](int a,int b){
        const float x=scores[a]+bias[a],y=scores[b]+bias[b]; return x==y?a<b:x>y;
    });
    float sum=0;
    for(int i=0;i<k;++i) sum+=scores[ids[i]];
    std::vector<Route> out;
    for(int i=0;i<k;++i) out.push_back({ids[i],scale*(scores[ids[i]]/(sum+1e-20f))});
    return out;
}
float swiglu(float gate,float up,float gc,float uc) {
    return (gc*std::tanh(gate/gc))*sigmoid(gate)*(uc*std::tanh(up/uc));
}

Checkpoint Checkpoint::open(const std::string& dir) {
    std::ifstream f(std::filesystem::path(dir)/"config.json",std::ios::binary);
    expect(f.good(),"missing config.json");
    std::string config(std::istreambuf_iterator<char>{f},{});
    Checkpoint c;
    c.config=parse_config(config);
    // The original release and requested fine-tune use this target shard;
    // do not enumerate the separate MTP shard as model weights.
    auto file=std::make_shared<SafeTensors>();
    std::string err;
    const auto path=(std::filesystem::path(dir)/"model-00001-of-00001.safetensors").string();
    if(!file->open(path,err)) throw std::runtime_error(err);
    c.layout=validate_tensors(c.config,file->tensors());
    c.file=std::move(file);
    return c;
}

ExpertMatrix Checkpoint::expert_matrix(int layer, int expert, bool down) const {
    expect(file && layer>=0 && layer<config.layers && size_t(layer)<layout.size() &&
           expert>=0 && expert<config.experts,"expert matrix index out of range");
    const std::string key="model.llm.layers."+std::to_string(layer)+".mlp.experts."+(down?"w2_weight":"w13_weight");
    auto data=[&](const std::string& suffix) {
        return static_cast<const uint8_t*>(file->data(tensor(file->tensors(),key+suffix)));
    };
    const auto fmt=down?layout[layer].down:layout[layer].gate_up;
    const int n=down?config.hidden:2*config.inter, k=down?config.inter:config.hidden;
    float outer=1.f;
    if(fmt==PackedFormat::NVFP4) {
        float amax, denom;
        std::memcpy(&amax,data("_amax")+size_t(expert)*4,4);
        std::memcpy(&denom,data("_outer_scale_denominator"),4);
        expect(std::isfinite(amax) && amax>=0 && std::isfinite(denom) && denom>0,"invalid NVFP4 outer scale");
        outer=amax/denom;
        expect(std::isfinite(outer),"NVFP4 outer scale overflows");
    }
    const size_t row=size_t(expert)*n;
    return {fmt,n,k,data("")+row*payload_row_bytes(fmt,k),data("_scale")+row*scale_row_bytes(fmt,k),outer};
}

} // namespace b70::inkling2
