// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/sampling.hpp"
namespace b70 {
// Sparse, normalized top-k/top-p distribution. IDs are target vocabulary IDs.
// The bounded support makes exact speculative correction cheap without
// transporting full vocabulary probabilities to the CPU.
struct SparseSamplingDistribution {
    static constexpr int capacity=64;
    int count=0,proposal=-1;
    uint64_t position=0,epoch=0,seed=0;
    int32_t ids[capacity]{};
    float probability[capacity]{};
};
struct SamplingDistributionView {
    int count;
    const int32_t* ids;
    const float* probability;
};
constexpr uint64_t kProposalStream=UINT64_C(0x36d6a3b1f5b47821);
constexpr uint64_t kAcceptanceStream=UINT64_C(0xa513cb6f19720d83);
constexpr uint64_t kResidualStream=UINT64_C(0x71be32d4c6f9085a);
template<class Distribution>
inline float sparse_probability(const Distribution& d,int id) {
    for(int j=0;j<d.count;++j)if(d.ids[j]==id)return d.probability[j];
    return 0;
}
template<class Distribution>
inline int sparse_draw(const Distribution& d,float uniform) {
    float sum=0;
    for(int j=0;j<d.count;++j){sum+=d.probability[j];if(uniform<sum)return d.ids[j];}
    return d.count?d.ids[d.count-1]:-1;
}
// Leviathan et al., ICML 2023, Algorithm 1: accept q's proposal with
// min(1,p/q), otherwise draw from normalized max(p-q,0). This preserves
// the target distribution while permitting more accepted drafts than
// independent target/proposal draws. No q-only token has positive residual.
template<class Target,class Proposal>
inline int coupled_sample(const Target& p,
                          const Proposal& q,
                          float accept_uniform,float residual_uniform,bool& accepted) {
    accepted=false;
    const float qp=sparse_probability(q,q.proposal);
    if(!(qp>0))return -1;
    const float pp=sparse_probability(p,q.proposal);
    if(accept_uniform*qp<pp){accepted=true;return q.proposal;}
    float total=0;
    for(int j=0;j<p.count;++j){
        const float difference=p.probability[j]-sparse_probability(q,p.ids[j]);
        total+=difference>0?difference:0;
    }
    if(!(total>0))return -1;
    const float draw=residual_uniform*total;float sum=0;
    int last=-1;
    for(int j=0;j<p.count;++j){
        const float difference=p.probability[j]-sparse_probability(q,p.ids[j]);
        if(difference>0){sum+=difference;last=p.ids[j];if(draw<sum)return last;}
    }
    if(last>=0)return last;
    return -1;
}
}
