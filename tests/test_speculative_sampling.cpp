// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/speculative_sampling.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
using namespace b70;
int main(){
    SparseSamplingDistribution p,q;
    p.count=q.count=3;
    for(int j=0;j<3;++j)p.ids[j]=q.ids[j]=17+13*j;
    p.probability[0]=.6f;p.probability[1]=.3f;p.probability[2]=.1f;
    q.probability[0]=.5f;q.probability[1]=.2f;q.probability[2]=.3f;
    constexpr int N=200000;int count[3]={};int accepted_count=0;
    for(int i=0;i<N;++i){
        q.proposal=sparse_draw(q,sampling_uniform_f32(123^kProposalStream,i));
        bool accepted=false;
        const int token=coupled_sample(p,q,sampling_uniform_f32(123^kAcceptanceStream,i),
            sampling_uniform_f32(123^kResidualStream,i),accepted);
        bool found=false;for(int j=0;j<3;++j)if(token==p.ids[j]){++count[j];found=true;}
        assert(found);accepted_count+=accepted;
        if(!accepted)assert(token!=q.proposal);
    }
    for(int j=0;j<3;++j)assert(std::abs(double(count[j])/N-p.probability[j])<.006);
    assert(std::abs(double(accepted_count)/N-.8)<.006);
    auto same=p;
    for(int i=0;i<1000;++i){
        same.proposal=sparse_draw(same,sampling_uniform_f32(5,i));bool accepted=false;
        assert(coupled_sample(p,same,sampling_uniform_f32(6,i),sampling_uniform_f32(7,i),accepted)==same.proposal);
        assert(accepted);
    }
    auto disjoint=q;for(int j=0;j<3;++j)disjoint.ids[j]+=1000;
    for(int i=0;i<1000;++i){
        disjoint.proposal=sparse_draw(disjoint,sampling_uniform_f32(5,i));bool accepted=true;
        const int token=coupled_sample(p,disjoint,sampling_uniform_f32(6,i),sampling_uniform_f32(7,i),accepted);
        assert(!accepted&&token!=disjoint.proposal&&sparse_probability(p,token)>0);
    }
    auto invalid=q;invalid.proposal=-1;bool accepted=true;
    assert(coupled_sample(p,invalid,.1f,.5f,accepted)==-1&&!accepted);
    std::cout<<"COUPLED_SAMPLING_LAW_PASS counts "<<count[0]<<" "<<count[1]<<" "<<count[2]<<" accepted "<<accepted_count<<"/"<<N<<"\n";
    // GRIMOIRE_SAMPLE_GUMBEL: the target's Gumbel-max draw over its kept set is
    // distributed as p, a drafter with the same weights and noise always agrees,
    // and a drafter with q agrees more often than a greedy draft is accepted.
    auto gumbel=[](uint64_t seed,uint64_t pos,int token){
        return -std::log(-std::log(double(sampling_gumbel_uniform(seed,pos,token))));};
    auto draw=[&](const SparseSamplingDistribution& d,uint64_t pos){
        int best=0;double value=-1e300;
        for(int j=0;j<d.count;++j){
            const double v=std::log(double(d.probability[j]))+gumbel(321,pos,d.ids[j]);
            if(v>value){value=v;best=j;}
        }
        return d.ids[best];};
    int gcount[3]={},same_draws=0,q_agree=0,greedy_accept=0;
    for(int i=0;i<N;++i){
        const int t=draw(p,uint64_t(i));
        for(int j=0;j<3;++j)if(t==p.ids[j])++gcount[j];
        same_draws+=draw(p,uint64_t(i))==t;q_agree+=draw(q,uint64_t(i))==t;greedy_accept+=t==p.ids[0];
    }
    for(int j=0;j<3;++j)assert(std::abs(double(gcount[j])/N-p.probability[j])<.006);
    assert(same_draws==N&&q_agree>greedy_accept);
    std::cout<<"GUMBEL_COUPLED_LAW_PASS counts "<<gcount[0]<<" "<<gcount[1]<<" "<<gcount[2]
             <<" q-drafter agrees "<<q_agree<<"/"<<N<<" greedy accepted "<<greedy_accept<<"/"<<N<<"\n";
}
