#include "b70/generation.hpp"
#include <cassert>
#include <iostream>

struct Engine {
    struct {int vocab=64;}cfg;
    int max_seq=64,pos=0,resets=0,forward_calls=0,draft_calls=0,commits=0;
    int reject_at=-1, batch_count=0, batch_verifies=0;
    int snapshots=0, restores=0;
    // decline_verify models the REAL contract: Grimoire::prefill returns
    // false only before submitting any work, so the caller may redo the
    // verify sequentially.  fail_forward is the genuine hard failure.
    bool graph_ok=false, decline_verify=false, fail_prompt=false;
    // fail the Nth forward onwards, so the PROMPT can succeed and the
    // failure lands inside the speculative verify where it belongs.
    int fail_forward_after=-1;
    std::vector<int32_t> history;
    float logits=0;
    void reset(){++resets;pos=0;history.clear();}
    void sync(){}
    int argmax_token(){assert(!history.empty());return (history.back()+1)%cfg.vocab;}
    const float* forward(int t){
        if(fail_forward_after>=0&&forward_calls>=fail_forward_after)return nullptr;
        assert(pos<max_seq);++forward_calls;history.push_back(t);++pos;return &logits;}
    bool prefill(const std::vector<int32_t>& ids,std::vector<int32_t>* next=nullptr){
        // No work submitted before declining -- that is the invariant the
        // sequential verify fallback in generation.hpp depends on.
        if(next&&decline_verify)return false;
        if(!next&&fail_prompt){fail_prompt=false;forward(ids[0]);return false;}
        for(int t:ids){forward(t);if(next)next->push_back(argmax_token());}
        if(next)++batch_verifies;
        batch_count=0;return true;
    }
    bool build_graph(){return false;}
    const float* step(){return forward(argmax_token());}
    bool recurrent=false;   // model has linear-attention layers
    bool has_recurrent_state()const{return recurrent;}
    void snapshot_recurrent(){++snapshots;}
    // Models the real thing: back to the pre-draft state, and the caller
    // replays the accepted tokens itself.
    void restore_recurrent(int saved){++restores;history.resize(size_t(saved));pos=saved;}
    void commit_spec_prefix(int saved,int n){++commits;history.resize(saved+n);pos=saved+n;}
    int mtp_draft(int t,int,bool){++draft_calls;return (t+1+(batch_count++==reject_at?7:0))%cfg.vocab;}
    bool dflash_draft(int t,int,std::vector<int32_t>& block,bool context=false){
        if(context)return true;
        for(int i=0;i<15;++i){t=(t+1+(i==reject_at?7:0))%cfg.vocab;block.push_back(t);}return true;
    }
};
template<class F> void throws(F f){bool did=false;try{f();}catch(const std::exception&){did=true;}assert(did);}
int main(){
    using namespace b70;
    std::vector<int32_t> out;FinishReason reason;
    const std::vector<int32_t> prompt{3,4,5,6};
    for(bool df:{false,true})for(int depth=0;depth<=15;++depth)for(int rejected=-1;rejected<depth;++rejected) {
        Engine e;e.reject_at=rejected;
        GenerationOptions o{40,-1,-1,depth,df,!df,false};
        const int n=generate_tokens(e,prompt,o,out,{},reason);
        assert(n==40&&reason==FinishReason::Length);
        for(int i=0;i<n;++i)assert(out[i]==i+7);
        assert(e.pos<=e.max_seq);
    }
    for(int capacity=5;capacity<=24;++capacity)for(int count=1;count<=40;++count) {
        Engine e;e.max_seq=capacity;
        GenerationOptions o{count,-1,-1,15,false,true,false};
        const int n=generate_tokens(e,prompt,o,out,{},reason);
        assert(n==std::min(count,capacity-4));assert(e.pos<=capacity);
    }
    Engine e;GenerationOptions o{5,-1,-1,3,false,true,false};
    generate_tokens(e,prompt,o,out,[](int){return false;},reason);
    assert(out.size()==1&&e.forward_calls==4&&e.draft_calls==0&&reason==FinishReason::Cancelled);
    Engine retry;retry.fail_prompt=true;
    o.mtp=false;generate_tokens(retry,prompt,o,out,{},reason);
    assert(retry.resets==2&&out==std::vector<int32_t>({7,8,9,10,11}));
    Engine zero;o.mtp=true;o.draft_depth=0;
    generate_tokens(zero,prompt,o,out,{},reason);assert(out.size()==5&&zero.draft_calls==0);
    // A batched verify that DECLINES must fall back to a token-at-a-time
    // verify and produce exactly what it would have produced anyway --
    // not throw.  Before the fallback existed, any engine without a
    // usable batched path failed every speculative request outright.
    {
        o.draft_depth=3;
        Engine plain;GenerationOptions po{20,-1,-1,0,false,false,false};
        std::vector<int32_t> want;
        generate_tokens(plain,prompt,po,want,{},reason);
        // rejected == -1 accepts everything, so nothing would ever roll
        // back; the point of this case is the ROLLBACK, so drive both.
        for(bool df:{false,true})for(int rejected:{-1,0,1,2}){
            Engine dec;dec.decline_verify=true;dec.reject_at=rejected;
            GenerationOptions so{20,-1,-1,3,df,!df,false};
            std::vector<int32_t> got;
            generate_tokens(dec,prompt,so,got,{},reason);
            assert(got==want);              // exact, whatever was rejected
            assert(dec.batch_verifies==0);  // the batched path really declined
            if(rejected>=0)assert(dec.commits>0);   // rollback happened
        }
    }
    // A RECURRENT model whose batched verify declines must not commit a
    // speculative prefix: the images commit_spec_prefix replays from are
    // written only by the batched path.  spec_verify_available() answers
    // for the device, not for the call, so this is reachable whenever a
    // scratch allocation fails -- and the old code walked straight into
    // it.  The round degrades to a plain step instead, which must still
    // produce exactly the non-speculative tokens, and must not be counted
    // as a speculative step because its draft was never judged.
    {
        Engine plain;GenerationOptions po{20,-1,-1,0,false,false,false};
        std::vector<int32_t> want;
        generate_tokens(plain,prompt,po,want,{},reason);
        for(bool df:{false,true})for(int rejected:{-1,0,1,2}){
            Engine rec;rec.recurrent=true;rec.decline_verify=true;rec.reject_at=rejected;
            SpecStats st;
            GenerationOptions so{20,-1,-1,3,df,!df,false,&st};
            std::vector<int32_t> got;
            generate_tokens(rec,prompt,so,got,{},reason);
            assert(got==want);            // still exactly plain decode
            assert(rec.commits==0);       // never committed an unsaved prefix
            assert(rec.batch_verifies==0);
            assert(st.steps==0);          // no round was ever judged
        }
    }
    // A verify whose forward actually fails is still a hard failure.
    Engine failure;failure.decline_verify=true;
    failure.fail_forward_after=int(prompt.size())+1;o.draft_depth=3;o.mtp=true;
    throws([&]{generate_tokens(failure,prompt,o,out,{},reason);});
    Engine one;o.max_tokens=1;o.mtp=false;
    generate_tokens(one,prompt,o,out,{},reason);assert(one.forward_calls==4);
    Engine eos;o.max_tokens=10;o.eos=9;
    generate_tokens(eos,prompt,o,out,{},reason);assert(out==std::vector<int32_t>({7,8})&&reason==FinishReason::Stop);
    // ---- speculation accounting -------------------------------------
    // An A/B between two drafters is only readable as accepted-per-step;
    // tok/s moves for two reasons at once.  Pin the arithmetic: the anchor
    // is never a draft, a rejected step still counts, and the totals match
    // what the fake engine was told to reject.
    for(bool df:{false,true})for(int depth=1;depth<=15;++depth)for(int rejected=-1;rejected<depth;++rejected) {
        Engine se;se.reject_at=rejected;
        SpecStats st;
        // max_tokens is a whole number of full rounds (anchor + depth
        // accepted), so with nothing rejected no round is truncated by the
        // budget and the totals are exact rather than bounded.
        const int rounds=3, budget=rounds*(depth+1);
        GenerationOptions so{budget,-1,-1,depth,df,!df,false,&st};
        generate_tokens(se,prompt,so,out,{},reason);
        assert(st.steps>0);
        assert(st.drafted<=st.steps*depth);            // anchor is never a draft
        assert(st.drafted>=st.steps);                  // every counted round drafted
        assert(st.accepted<=st.drafted);
        if(rejected<0) {                               // verifier agrees throughout
            assert(st.steps==rounds);
            assert(st.drafted==st.steps*depth);
            assert(st.accepted==st.drafted);
        } else {
            // The first mismatch is at index `rejected`, so a round that
            // drafted its full depth contributes exactly that many.
            assert(st.accepted<=st.steps*rejected);
        }
        // MTP drafts one token per call, so the engine's own call count is
        // an independent check that nothing is double-counted.
        if(!df)assert(st.drafted==se.draft_calls);
        assert(st.per_step()==double(st.accepted)/double(st.steps));
    }
    {   // A request cancelled part-way through emitting a round's accepted
        // tokens must still count that round: the draft was proposed and
        // judged.  Dropping it would bias the rate upward on exactly the
        // requests that end early.  (Cancelling on the very first token
        // happens before any draft, so accept the anchor and refuse the
        // next one.)
        Engine ce;SpecStats st;int seen=0;
        GenerationOptions co{40,-1,-1,4,false,true,false,&st};
        generate_tokens(ce,prompt,co,out,[&](int){return ++seen<2;},reason);
        assert(reason==FinishReason::Cancelled);
        assert(st.steps==1&&st.drafted==4&&st.accepted==4);
    }
    {   // no stats pointer means no accounting and no crash
        Engine ne;GenerationOptions no{8,-1,-1,4,false,true,false};
        assert(no.stats==nullptr);
        generate_tokens(ne,prompt,no,out,{},reason);
    }
    throws([&]{generate_tokens(e,{},o,out,{},reason);});
    throws([&]{generate_tokens(e,{-1},o,out,{},reason);});
    throws([&]{generate_tokens(e,{64},o,out,{},reason);});
    throws([&]{generate_tokens(e,std::vector<int32_t>(64,1),o,out,{},reason);});
    std::cout<<"generation: greedy parity for every rejection prefix, capacity, cancellation and errors PASS\n";
}
