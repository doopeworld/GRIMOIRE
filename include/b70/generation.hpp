#pragma once
#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace b70 {
enum class FinishReason { Stop, Length, Cancelled };
inline const char* finish_reason_name(FinishReason r) {
    return r==FinishReason::Stop?"stop":(r==FinishReason::Length?"length":"cancelled");
}
// Speculation accounting.  tok/s alone cannot settle any question about a
// drafter: it moves when the draft gets cheaper AND when it gets more
// accurate, and those want opposite decisions.  accepted/steps is the half
// that isolates proposal quality, so any A/B between two draft
// configurations -- MTP vs DFlash2, selector on vs off, one K-norm
// convention vs another -- needs it recorded, not inferred.
struct SpecStats {
    long long steps=0;      // speculative rounds entered
    long long drafted=0;    // draft tokens proposed (anchor excluded)
    long long accepted=0;   // draft tokens the verifier kept
    double per_step() const { return steps ? double(accepted)/double(steps) : 0.0; }
    double rate()     const { return drafted ? double(accepted)/double(drafted) : 0.0; }
};
struct GenerationOptions {
    int max_tokens=0, eos=-1, eot=-1, draft_depth=0;
    bool dflash=false, mtp=false, graph=false;
    // Optional; nothing is recorded when null.  Declared last so the
    // existing aggregate initialisations stay valid.
    SpecStats* stats=nullptr;
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
    // Snapshot at the END of the request, covering prompt AND reply, so
    // the next turn can resume from the whole conversation rather than
    // from the prompt this one happened to start with.  Without it the
    // snapshot never grows and reuse covers only the first turn forever.
    //
    // RAII because this function returns from six places (every emit()
    // that stops) and can throw from more.  Rule 7c's argument: the paths
    // that need it most have no exit to annotate.
    //
    // NOT on the exception path: a request that threw left the engine
    // mid-flight, and a snapshot of that state would be handed to the
    // next request as though it described a finished one.
    struct PrefixSnapshot {
        Engine& e; const std::vector<int32_t>& prompt;
        const std::vector<int32_t>& out;
        ~PrefixSnapshot() {
            if(std::uncaught_exceptions())return;
            std::vector<int32_t> all=prompt;
            all.insert(all.end(),out.begin(),out.end());
            // TRUNCATE TO WHAT THE ENGINE ACTUALLY PROCESSED.  emit()
            // pushes a token BEFORE forward() consumes it, so the last
            // token of `out` has been returned to the caller and never
            // fed back in -- the cache holds one entry fewer than this
            // list names.  Claiming otherwise resumes at a position whose
            // KV row was never written, and the answer that comes out is
            // fluent and slightly wrong, which took a three-turn gate to
            // see (turns one and two matched).
            //
            // e.pos is the count the engine will stand behind, so deriving
            // the length from it cannot drift from the loop's shape again.
            if(e.pos>=0&&size_t(e.pos)<all.size()){
                all.resize(size_t(e.pos));
            } else if(e.pos>=0&&size_t(e.pos)>all.size()){
                // THE OTHER DIRECTION (external audit F4, 2026-09-21).
                // Speculative verification commits a WHOLE accepted
                // block to the recurrent state in one call
                // (commit_spec_prefix) BEFORE the loop that delivers
                // those tokens to `out` one at a time even runs -- so if
                // that delivery loop stops partway (EOS inside the
                // accepted block, or a cancelled callback), e.pos
                // already reflects tokens this snapshot's own token list
                // does not name. That is the OPPOSITE of the case above:
                // here the cache holds MORE than this list admits to,
                // not fewer, and there is no direction to resize `all`
                // that fixes it -- shrinking would lose real,
                // already-committed tokens from the label, and there is
                // nothing to grow it WITH, since what actually got
                // emitted to the caller is exactly `all` already.
                //
                // A restore later would set pos = all.size() (the
                // label) while the copied recurrent state is however
                // many extra tokens further along -- invisible for a
                // plain KV cache (rows past the labeled position are
                // simply never read) but real corruption for a
                // hybrid/DeltaNet model, whose recurrent state is a
                // single rolling summary with no per-position rollback.
                // Restore-and-replay to repair it is not an option: it
                // is documented elsewhere in this engine as corrupting
                // memory on exactly these models.
                //
                // So: decline to cache this termination rather than
                // cache a label that lies about what state it names.
                // Losing one turn's resume is a re-read next time;
                // saving a mislabeled snapshot is silent wrong output on
                // every turn after that, on the one architecture family
                // where nothing else would catch it.
                return;
            }
            e.save_prefix_now(all);
        }
    } prefix_snapshot{e,prompt,out};
    // REUSE WHAT IS ALREADY RESIDENT.  A chat turn is the previous prompt
    // plus a reply plus a new message, so the snapshot taken at the end of
    // the last request is a strict PREFIX of this one.  Before this,
    // reset() threw that away and the whole history was re-read on every
    // single turn -- the cache only fired on a byte-identical repeat,
    // which a growing conversation never produces.
    //
    // reuse == 0 is the old path exactly: reset, prefill everything.
    const int reuse = e.prefix_reuse(prompt);
    bool resumed = false;
    if(reuse>0) {
        e.sync();
        resumed = e.restore_prefix_upto(reuse);
    }
    if(!resumed) e.reset();
    // The tail is what is left to process.  It is never empty: prefix_reuse
    // keeps a token back precisely so prefill still produces this request's
    // logits rather than leaving the previous request's in place.
    const std::vector<int32_t> tail = resumed
        ? std::vector<int32_t>(prompt.begin()+reuse, prompt.end()) : prompt;
    if(!e.prefill(tail)) {
        // A rejected batch may already have submitted work. Drain before a
        // sequential retry so no token is processed twice.  A resumed
        // request must NOT reset here -- that would discard the very
        // prefix it just restored and then replay only the tail, which
        // silently answers from a truncated conversation.
        e.sync();
        if(resumed) {
            if(!e.restore_prefix_upto(reuse))
                throw std::runtime_error("prefix restore failed on retry");
        } else e.reset();
        for(auto t:tail)if(!e.forward(t))throw std::runtime_error("sequential prefill failed");
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
            bool degraded=false;   // draft discarded unjudged, see below
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
            if(!e.prefill(candidates,&verified)) {
                // The batched verify declined.  prefill() returns false
                // only BEFORE it submits any work -- an unavailable
                // batched path, or scratch it could not allocate -- so
                // the engine state is untouched and the same verify can
                // be done a token at a time.  Without this, a degraded
                // batched path turned every speculative request into a
                // hard failure: correct output was available and the
                // request threw anyway.
                verified.clear();
                if(e.has_recurrent_state()) {
                    // A sequential verify on a recurrent model cannot be
                    // rolled back: commit_spec_prefix replays from the
                    // per-step images only the BATCHED verify writes, and
                    // restore-and-replay corrupts memory (see below).  It
                    // is not enough that the engine said a batched verify
                    // was AVAILABLE -- spec_verify_available() answers for
                    // the device, not for this call, and a scratch
                    // allocation can still fail here.  Nothing has been
                    // submitted yet, so degrade this round to a plain
                    // step: forward the verified anchor alone and leave
                    // the drafted suffix unused.
                    degraded = true;
                    candidates.resize(1);
                    if(!e.forward(candidates[0]))
                        throw std::runtime_error("speculative verification failed");
                    verified.push_back(e.argmax_token());
                } else {
                    for(size_t i=0;i<candidates.size();++i){
                        if(!e.forward(candidates[i]))
                            throw std::runtime_error("speculative verification failed");
                        verified.push_back(e.argmax_token());
                    }
                }
            }
            if(verified.size()!=candidates.size())
                throw std::runtime_error("speculative verification failed");
            int accepted=1;
            while(accepted<int(candidates.size())&&candidates[accepted]==verified[accepted-1])++accepted;
            // Counted BEFORE the emit loop below, which can return early on
            // a stop token or a cancelled callback: the round happened and
            // its proposals were judged either way, and dropping it would
            // bias the rate upward on exactly the requests that end early.
            // A degraded round proposed a draft and then threw it away
            // without the verifier ever judging it.  Counting it as a step
            // with zero accepted would report an acceptance rate for a
            // round that never happened.
            if(o.stats && !degraded) {
                ++o.stats->steps;
                o.stats->drafted  += int(candidates.size())-1;   // anchor is not a draft
                o.stats->accepted += accepted-1;
            }
            if(accepted<int(candidates.size())) {
                // commit_spec_prefix is exact whenever there is no
                // recurrent state to rebuild, whichever verify ran.  When
                // there IS recurrent state, only the batched verify writes
                // the per-step images it replays from -- and the engine
                // refuses to speculate at all in that case
                // (spec_verify_available), so this is never reached with a
                // sequential verify and recurrent state.
                //
                // A restore-and-replay rollback was tried here and does NOT
                // work: restore_recurrent alone is clean, but re-running
                // forward() at a position that has already been processed
                // corrupts memory on a DeltaNet model -- reproducible, and
                // it surfaces as a SIGSEGV at teardown because USM on a CPU
                // device is host malloc.  That is a latent bug in the
                // linear-attention decode path worth chasing on the card;
                // until it is, do not build a rollback on re-entering a
                // position.
                e.commit_spec_prefix(saved,accepted);
            }
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
