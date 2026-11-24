// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/inkling2.hpp"
#include "b70/formats.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace b70::inkling2 {
namespace {
using Row=std::vector<float>;
using In=std::span<const float>;
constexpr const char* root="model.llm.";
float bf(float x) { return bf16_to_f32(f32_to_bf16(x)); }
float sig(float x) { return x>=0 ? 1.f/(1.f+std::exp(-x)) : std::exp(x)/(1.f+std::exp(x)); }
struct Bf {
    const uint8_t* p;
    float operator[](size_t i) const {
        return bf16_to_f32({uint16_t(uint16_t(p[2*i])|(uint16_t(p[2*i+1])<<8))});
    }
};
struct F32 {
    const uint8_t* p;
    float operator[](size_t i) const { float x; std::memcpy(&x,p+4*i,4); return x; }
};
float dot(In a,In b) {
    float s=0;
    for(size_t i=0;i<a.size();++i) s+=a[i]*b[i];
    return s;
}
float dot(In a,Bf b,size_t off) {
    float s=0;
    for(size_t i=0;i<a.size();++i) s+=a[i]*b[off+i];
    return s;
}
Row dense(In x,Bf w,int outputs,bool round=true) {
    Row y(outputs);
    for(int i=0;i<outputs;++i) {
        const float s=dot(x,w,size_t(i)*x.size());
        y[i]=round?bf(s):s;
    }
    return y;
}
float rms_inv(In x,float eps) { return 1.f/std::sqrt(dot(x,x)/float(x.size())+eps); }
Row rms(In x,Bf w,float eps) {
    Row y(x.size()); const float r=rms_inv(x,eps);
    for(size_t i=0;i<x.size();++i) y[i]=bf((x[i]*r)*w[i]);
    return y;
}
void clip(Row& v,int heads,int dim,float threshold,const Bf* scale,float tau) {
    for(int h=0;h<heads;++h) {
        auto* p=v.data()+h*dim;
        float s=0;
        for(int d=0;d<dim;++d) { if(scale) p[d]*=(*scale)[h*dim+d]; s+=p[d]*p[d]; }
        const float f=std::min(1.f,threshold/(std::sqrt(s)+1e-6f));
        for(int d=0;d<dim;++d) p[d]=bf(bf(p[d]*f)*tau);
    }
}
// Both the attention scores and the mixed values use normalized residual rows.
Row mix(const std::vector<Row>& bank,In partial,F32 query,size_t query_off) {
    std::vector<float> scores, rs;
    scores.reserve(bank.size()+1); rs.reserve(bank.size()+1);
    const float f=1.f/std::sqrt(float(partial.size()));
    float highest=-std::numeric_limits<float>::infinity();
    auto visit=[&](In row) {
        const float inv=rms_inv(row,1e-6f); float score=0;
        for(size_t i=0;i<row.size();++i) score+=query[query_off+i]*(row[i]*inv);
        score*=f;
        highest=std::max(highest,score); scores.push_back(score); rs.push_back(inv);
    };
    for(auto& x:bank) visit(x);
    visit(partial);
    float sum=0;
    for(float& x:scores) { x=std::exp(x-highest); sum+=x; }
    Row out(partial.size(),0);
    for(size_t j=0;j<scores.size();++j) {
        const In x=j<bank.size()?In(bank[j]):partial;
        const float w=scores[j]/sum;
        for(size_t i=0;i<x.size();++i) out[i]+=(x[i]*rs[j])*w;
    }
    return out;
}

struct Slot {
    Row key,value;
    std::array<Row,4> history;
    int capacity=0;
};
Row convolve(In x,Row& buf,int pos,int taps,Bf stat,In dyn={},int head_dim=1) {
    const int n=int(x.size());
    if(buf.empty()) buf.resize(size_t(n)*(taps-1));
    const float scale=1.f/std::sqrt(float(1+taps*(dyn.empty()?1:2)));
    Row out(n);
    for(int ch=0;ch<n;++ch) {
        float sum=0;
        for(int lag=0;lag<=std::min(pos,taps-1);++lag) {
            const float xlag=lag==0?x[ch]:buf[size_t((pos-lag)%(taps-1))*n+ch];
            // The static-only file is chronological [channel,1,tap].
            // For dynamic K/V the file is [lag,channel] with lag 0 current.
            const float w=dyn.empty()?stat[ch*taps+taps-1-lag]:stat[lag*n+ch]+dyn[ch/head_dim*taps+lag];
            sum+=xlag*w;
        }
        out[ch]=bf((x[ch]+sum)*scale);
    }
    std::copy(x.begin(),x.end(),buf.begin()+size_t(pos%(taps-1))*n);
    return out;
}
} // namespace

struct TextReference::Impl {
    Checkpoint ck;
    std::vector<Visit> visits;
    int limit,pos=0;
    bool valid=true;
    std::vector<Slot> slots;
    explicit Impl(Checkpoint w,int context):ck(std::move(w)),visits(ck.config.visits()),limit(context),slots(ck.config.cache_slots()) {
        if(limit<1 || limit>ck.config.max_context || !ck.file || ck.layout.size()!=size_t(ck.config.layers))
            throw std::invalid_argument("invalid Inkling2 reference context or checkpoint");
    }
    const uint8_t* bytes(const std::string& name) const {
        const auto* t=ck.file->find(name);
        if(!t) throw std::runtime_error("Inkling2 tensor disappeared: "+name);
        return static_cast<const uint8_t*>(ck.file->data(*t));
    }
    Bf w(const std::string& k) const { return Bf{bytes(k)}; }
    F32 f(const std::string& k) const { return F32{bytes(k)}; }
    const Config& c() const { return ck.config; }
    Row gate(In cur,In message,const std::string& prefix) {
        const int h=c().hidden;
        const float ir=rms_inv(cur,c().eps),im=rms_inv(message,c().eps);
        Row x(2*h);
        for(int i=0;i<h;++i) { x[i]=bf(cur[i]*ir); x[h+i]=bf(message[i]*im); }
        auto low=dense(x,w(prefix+"down_weight"),c().gate_rank);
        // The checkpoint stores BF16 parameters; the high projections in the
        // reference are widened for FP32 tanh and the residual blend.
        auto gain=dense(low,w(prefix+"gain_weight"),h,false);
        auto forget=dense(low,w(prefix+"forget_weight"),h,false);
        Row out(h);
        for(int i=0;i<h;++i) out[i]=(1.f-std::tanh(forget[i]))*cur[i]+std::tanh(gain[i])*message[i];
        return out;
    }
    PackedRows packed(const std::string& name,int rows,int cols,PackedFormat type,Row& amax) {
        PackedRows p{type,c().experts,rows,cols,bytes(name),bytes(name+"_scale")};
        if(type==PackedFormat::NVFP4) {
            amax.resize(c().experts); const auto a=f(name+"_amax");
            for(int e=0;e<c().experts;++e) amax[e]=a[e];
            p.amax=amax.data(); p.denominator=f(name+"_outer_scale_denominator")[0];
        }
        return p;
    }
    Row moe(In x,const std::string& l,int physical) {
        const int h=c().hidden,I=c().inter,E=c().experts;
        const auto logit=dense(x,w(l+"mlp.gate.weight"),E,false);
        Row bias(E); const auto b=f(l+"mlp.gate.bias");
        for(int e=0;e<E;++e) bias[e]=b[e];
        auto selected=route(logit,bias,c().top_k,c().route_scale);
        // Match the eager expert accumulation order; routing scores are computed separately.
        std::sort(selected.begin(),selected.end(),[](Route a,Route b){return a.expert<b.expert;});
        Row a13,a2;
        const auto layout=ck.layout[physical];
        auto gu=packed(l+"mlp.experts.w13_weight",2*I,h,layout.gate_up,a13);
        auto down=packed(l+"mlp.experts.w2_weight",h,I,layout.down,a2);
        Row weights(std::max(h,I)),act(I),out(h,0);
        auto eval=[&](const PackedRows& p,int e,int row,In a) {
            auto decoded=std::span(weights).first(a.size());
            p.row(e,row,decoded);
            float s=0;
            for(size_t i=0;i<a.size();++i) s+=a[i]*bf(decoded[i]);
            return bf(s);
        };
        for(auto [id,weight]:selected) {
            for(int i=0;i<I;++i) {
                const float g=eval(gu,id,2*i,x), u=eval(gu,id,2*i+1,x);
                act[i]=bf(swiglu(g,u,c().gate_cap,c().up_cap));
            }
            for(int j=0;j<h;++j) out[j]+=weight*eval(down,id,j,act);
        }
        for(auto& z:out) z=bf(z);
        return out;
    }
    Row attn(In xq,In xk,In xv,const std::string& l,const Visit& v) {
        auto& s=slots[v.cache_slot]; const auto a=v.sliding?c().local:c().full;
        const int D=a.head_dim,Q=a.heads*D,K=a.kv_heads*D,R=c().rel_dim, E=v.sliding?c().window:c().rel_extent;
        if(!s.capacity) {
            s.capacity=v.sliding?std::min(limit,c().window):limit;
            s.key.resize(size_t(s.capacity)*K); s.value.resize(size_t(s.capacity)*K);
        }
        const auto t=l+"attn.";
        Row q=dense(xq,w(t+"wq_du.weight"),Q);
        Row k=dense(xk,w(t+"wk_dv.weight"),K), val=dense(xv,w(t+"wv_dv.weight"),K);
        auto apply_conv=[&](Row z,In src,int slot,const std::string& n) {
            if(v.sliding) return convolve(z,s.history[slot],pos,c().conv,w(t+n+"_sconv.weight"));
            auto dynamic=dense(src,w(t+n+"_sconv.weight_proj"),a.kv_heads*c().conv);
            return convolve(z,s.history[slot],pos,c().conv,w(t+n+"_sconv.static_w"),dynamic,D);
        };
        k=apply_conv(std::move(k),xk,0,"k"); val=apply_conv(std::move(val),xv,1,"v");
        const float tau=v.sliding?1.f:1.f+c().log_alpha*std::log(std::max(1.f,float(pos+1)/c().log_floor));
        const Bf qs=w(t+"q_l2_clip.scale"),rs=w(t+"rel_l2_clip.scale");
        clip(q,a.heads,D,c().qk_clip,&qs,tau);
        clip(k,a.kv_heads,D,c().qk_clip,nullptr,1.f);
        for(int j=0;j<a.kv_heads;++j) {
            auto row=std::span(val).subspan(j*D,D);
            const float f=rms_inv(row,c().eps);
            for(auto& z:row) z=bf(z*f);
        }
        const size_t offset=size_t(pos%s.capacity)*K;
        std::copy(k.begin(),k.end(),s.key.begin()+offset); std::copy(val.begin(),val.end(),s.value.begin()+offset);
        Row rel=dense(xq,w(t+"wr_du.weight"),a.heads*R);
        clip(rel,a.heads,R,c().rel_clip,&rs,tau);
        const auto table=w(t+"rel_logits_proj.proj");
        const auto sinks=dense(xq,w(t+"attn_sink_proj.weight"),a.heads);
        const auto g=dense(xq,w(t+"gate_attn.weight"),Q);
        const int count=std::min(pos+1,s.capacity),first=pos+1-count;
        Row scores(count),out(Q,0);
        const float scale=1.f/std::sqrt(float(D));
        const int groups=a.heads/a.kv_heads;
        for(int j=0;j<a.heads;++j) {
            const auto query=In(q).subspan(j*D,D);
            const int hd=j/groups*D;
            float high=sinks[j];
            for(int i=0;i<count;++i) {
                const size_t p=size_t((first+i)%s.capacity)*K+hd;
                const int distance=pos-(first+i);
                float bias=0;
                if(distance<E) for(int r=0;r<R;++r) bias+=rel[j*R+r]*table[r*E+distance];
                const float qk=bf(bf(dot(query,In(s.key).subspan(p,D)))*scale);
                scores[i]=bf(qk+bf(bias)); high=std::max(high,scores[i]);
            }
            float sum=std::exp(sinks[j]-high);  // sink absorbs probability, but has zero value
            for(auto& z:scores) { z=std::exp(z-high); sum+=z; }
            for(int i=0;i<count;++i) {
                const size_t p=size_t((first+i)%s.capacity)*K+hd;
                const float weight=bf(scores[i]/sum);
                for(int d=0;d<D;++d) out[j*D+d]+=weight*s.value[p+d];
            }
            for(int d=0;d<D;++d) out[j*D+d]=bf(bf(out[j*D+d])*sig(g[j*D+d]));
        }
        return dense(out,w(t+"wo_ud.weight"),c().hidden);
    }
    int step(int token,std::span<float> logits) {
        const auto& cfg=c(); const int h=cfg.hidden;
        if(!valid || pos>=limit || token<0 || token>=cfg.output_vocab || (!logits.empty() && logits.size()!=size_t(cfg.output_vocab)))
            throw std::invalid_argument("invalid Inkling2 token, position or logits buffer (reset after an execution error)");
        // If a read/math failure happens after appending state, refuse to continue at the old position.
        valid=false;
        const auto embed=w(std::string(root)+"embed.weight");
        Row hidden(h); for(int i=0;i<h;++i) hidden[i]=embed[size_t(token)*h+i];
        hidden=rms(hidden,w(std::string(root)+"embed_norm.weight"),cfg.eps);
        std::vector<Row> bank,pre_bank,previous_bank;
        Row pre_hidden,previous_hidden;
        for(auto v:visits) {
            const int i=v.physical;
            if(i==cfg.loop_start && v.pass==0) { pre_hidden=hidden; pre_bank=bank; }
            if(i==cfg.loop_start && v.has_message) {
                previous_hidden=hidden;
                previous_bank.assign(bank.begin()+pre_bank.size()+1,bank.end());
                hidden=pre_hidden; bank=pre_bank;
            }
            const std::string prefix=std::string(root)+"layers."+std::to_string(i)+".";
            std::array<Row,4> msg;
            if(v.has_message) {
                const size_t off=size_t(i-cfg.loop_start)*2*h;
                const auto q=f(std::string(root)+"middle_cycle_cross_pass_queries");
                const auto kv=f(std::string(root)+"middle_cycle_cross_pass_kv_queries");
                msg={mix(previous_bank,previous_hidden,q,off),mix(previous_bank,previous_hidden,kv,off),
                     mix(previous_bank,previous_hidden,kv,off+h),mix(previous_bank,previous_hidden,q,off+h)};
            }
            const auto aquery=i?f(std::string(root)+"attn_res.attn_queries."+std::to_string(i-1)):F32{nullptr};
            std::array<Row,3> ready;
            for(int j=0;j<3;++j) {
                Row x=i?mix(bank,hidden,aquery,v.sliding?0:size_t(j)*h):hidden;
                if(v.has_message) {
                    const char* n=j==0?"attn":(j==1?"k":"v");
                    x=gate(x,msg[j],prefix+"controllers."+n+".");
                }
                const char* norm=v.sliding?"attn_norm.weight":(j==0?"q_input_norm.weight":(j==1?"k_input_norm.weight":"v_input_norm.weight"));
                ready[j]=rms(x,w(prefix+norm),cfg.eps);
            }
            auto z=attn(ready[0],ready[1],ready[2],prefix,v);
            auto& slot=slots[v.cache_slot];
            auto residual=convolve(z,slot.history[2],pos,cfg.conv,w(prefix+"attn_sconv.weight"));
            if(v.block_boundary) bank.push_back(hidden);
            else for(int j=0;j<h;++j) residual[j]+=hidden[j];
            Row x=mix(bank,residual,f(std::string(root)+"attn_res.mlp_queries."+std::to_string(i)),0);
            if(v.has_message) x=gate(x,msg[3],prefix+"controllers.mlp.");
            auto y=moe(rms(x,w(prefix+"mlp_norm.weight"),cfg.eps),prefix,i);
            y=convolve(y,slot.history[3],pos,cfg.conv,w(prefix+"mlp_sconv.weight"));
            for(int j=0;j<h;++j) hidden[j]=bf(residual[j]+y[j]);
        }
        auto x=rms(mix(bank,hidden,f(std::string(root)+"attn_res.output_query"),0),w(std::string(root)+"norm.weight"),cfg.eps);
        const auto unembed=w(std::string(root)+"unembed.weight");
        float high=-std::numeric_limits<float>::infinity(); int id=-1;
        for(int j=0;j<cfg.output_vocab;++j) {
            const float linear=bf(dot(x,unembed,size_t(j)*h));
            const float s=cfg.logit_cap*std::tanh(linear/cfg.logit_cap);
            if(!std::isfinite(s)) throw std::runtime_error("Inkling2 nonfinite unembed result");
            if(!logits.empty()) logits[j]=s;
            if(s>high){ high=s; id=j; }
        }
        ++pos; valid=true;
        return id;
    }
};

TextReference::TextReference(Checkpoint w,int c):p_(std::make_unique<Impl>(std::move(w),c)) {}
TextReference::~TextReference()=default;
TextReference::TextReference(TextReference&&) noexcept=default;
TextReference& TextReference::operator=(TextReference&&) noexcept=default;
void TextReference::reset() { p_->slots.assign(p_->c().cache_slots(),Slot{}); p_->pos=0; p_->valid=true; }
int TextReference::position() const { return p_->pos; }
const Config& TextReference::config() const { return p_->c(); }
int TextReference::step(int token,std::span<float> logits) { return p_->step(token,logits); }

} // namespace b70::inkling2
