#pragma once
#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace b70 {
enum class FinishReason { Stop, Length, Cancelled };
inline const char* finish_reason_name(FinishReason r) {
    return r==FinishReason::Stop?"stop":(r==FinishReason::Length?"length":"cancelled");
}
struct GenerationOptions {
    int max_tokens=0, eos=-1, eot=-1, draft_depth=0;
    bool dflash=false, mtp=false, graph=false;
};
inline int generation_budget(const std::vector<int32_t>& ids, int requested, int capacity, int vocab) {
    if(ids.empty())throw std::invalid_argument("prompt must contain at least one token");
    if(capacity<=0 || ids.size()>=size_t(capacity))
        throw std::invalid_argument("prompt leaves no room for completion in the context window");
    if(requested<0)throw std::invalid_argument("max_tokens must be nonnegative");
    for(auto t:ids)if(t<0||t>=vocab)throw std::invalid_argument("prompt token outside model vocabulary");
    return std::min(requested,capacity-int(ids.size()));
}

// Production control flow is a template to allow stateful, deterministic host
// tests without emulating GPU numerics. Errors propagate, never masquerade as EOS.
template<class Engine>
int generate_tokens(Engine& e, const std::vector<int32_t>& prompt,
                    GenerationOptions o, std::vector<int32_t>& out,
                    const std::function<bool(int32_t)>& callback,
                    FinishReason& reason) {
    out.clear();reason=FinishReason::Length;
    o.max_tokens=generation_budget(prompt,o.max_tokens,e.max_seq,e.cfg.vocab);
    if(!o.max_tokens)return 0;
    e.reset();
    if(!e.prefill(prompt)) {
        // A rejected batch may already have submitted work. Drain and reset
        // before a sequential retry so no token is processed twice.
        e.sync();e.reset();
        for(auto t:prompt)if(!e.forward(t))throw std::runtime_error("sequential prefill failed");
    }
    e.sync();
    if(o.dflash) {
        std::vector<int32_t> unused;
        if(!e.dflash_draft(0,e.pos,unused,true))throw std::runtime_error("DFlash context preparation failed");
    }
    if(o.graph && !o.dflash && !o.mtp && !e.graph_ok)e.build_graph();
    int tok=e.argmax_token();
    auto stop=[&](int t){return (o.eos>=0&&t==o.eos)||(o.eot>=0&&t==o.eot);};
    auto emit=[&](int t) {
        if(t<0||t>=e.cfg.vocab)throw std::runtime_error("engine returned an invalid token");
        if(stop(t)){reason=FinishReason::Stop;return false;}
        out.push_back(t);
        if(callback&&!callback(t)){reason=FinishReason::Cancelled;return false;}
        return int(out.size())<o.max_tokens;
    };
    while(int(out.size())<o.max_tokens) {
        // The verified anchor is immediately available, including at TTFT.
        if(!emit(tok))break;
        const int k=std::min({o.draft_depth,e.max_seq-e.pos-1,o.max_tokens-int(out.size())});
        if((o.dflash||o.mtp)&&k>0) {
            const int saved=e.pos;
            e.snapshot_recurrent();
            std::vector<int32_t> candidates{tok};
            if(o.dflash) {
                std::vector<int32_t> block;
                if(!e.dflash_draft(tok,saved,block)||block.empty())
                    throw std::runtime_error("DFlash draft failed");
                candidates.insert(candidates.end(),block.begin(),block.begin()+std::min(k,int(block.size())));
            } else {
                int d=tok;
                for(int j=1;j<=k;++j) {
                    d=e.mtp_draft(d,saved+j-1,j>1);
                    if(d<0||d>=e.cfg.vocab)throw std::runtime_error("MTP draft failed");
                    candidates.push_back(d);
                }
            }
            for(int t:candidates)if(t<0||t>=e.cfg.vocab)throw std::runtime_error("invalid draft token");
            std::vector<int32_t> verified;
            if(!e.prefill(candidates,&verified)||verified.size()!=candidates.size())
                throw std::runtime_error("speculative verification failed");
            int accepted=1;
            while(accepted<int(candidates.size())&&candidates[accepted]==verified[accepted-1])++accepted;
            if(accepted<int(candidates.size()))e.commit_spec_prefix(saved,accepted);
            for(int i=1;i<accepted;++i)if(!emit(candidates[i]))return int(out.size());
            tok=verified[accepted-1];
        } else {
            if(e.pos>=e.max_seq)break;
            if(!(e.graph_ok?e.step():e.forward(tok)))throw std::runtime_error("decode step failed");
            tok=e.argmax_token();
        }
    }
    return int(out.size());
}
} // namespace b70
