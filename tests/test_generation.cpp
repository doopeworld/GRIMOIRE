#include "b70/generation.hpp"
#include <cassert>
#include <iostream>

struct Engine {
    struct {int vocab=64;}cfg;
    int max_seq=64,pos=0,resets=0,forward_calls=0,draft_calls=0,commits=0;
    int reject_at=-1, batch_count=0;
    bool graph_ok=false, fail_verify=false, fail_prompt=false;
    std::vector<int32_t> history;
    float logits=0;
    void reset(){++resets;pos=0;history.clear();}
    void sync(){}
    int argmax_token(){assert(!history.empty());return (history.back()+1)%cfg.vocab;}
    const float* forward(int t){assert(pos<max_seq);++forward_calls;history.push_back(t);++pos;return &logits;}
    bool prefill(const std::vector<int32_t>& ids,std::vector<int32_t>* next=nullptr){
        if(next&&fail_verify){forward(ids[0]);return false;}
        if(!next&&fail_prompt){fail_prompt=false;forward(ids[0]);return false;}
        for(int t:ids){forward(t);if(next)next->push_back(argmax_token());}
        batch_count=0;return true;
    }
    bool build_graph(){return false;}
    const float* step(){return forward(argmax_token());}
    void snapshot_recurrent(){}
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
    Engine failure;failure.fail_verify=true;o.draft_depth=3;
    throws([&]{generate_tokens(failure,prompt,o,out,{},reason);});
    Engine one;o.max_tokens=1;o.mtp=false;
    generate_tokens(one,prompt,o,out,{},reason);assert(one.forward_calls==4);
    Engine eos;o.max_tokens=10;o.eos=9;
    generate_tokens(eos,prompt,o,out,{},reason);assert(out==std::vector<int32_t>({7,8})&&reason==FinishReason::Stop);
    throws([&]{generate_tokens(e,{},o,out,{},reason);});
    throws([&]{generate_tokens(e,{-1},o,out,{},reason);});
    throws([&]{generate_tokens(e,{64},o,out,{},reason);});
    throws([&]{generate_tokens(e,std::vector<int32_t>(64,1),o,out,{},reason);});
    std::cout<<"generation: greedy parity for every rejection prefix, capacity, cancellation and errors PASS\n";
}
