// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "b70/sampling.hpp"
#include <sycl/sycl.hpp>
#include <limits>
namespace b70 {
// Two bounded top-k reductions. No full-vocabulary device-to-host copy and
// no mutable RNG state inside a captured graph. Workspace belongs to one
// engine queue; in-order submissions permit reuse between verification rows.
struct SamplingWorkspace {
    static constexpr int groups=128, width=128, capacity=64;
    float* values=nullptr;
    int32_t* indices=nullptr;
    void ensure(sycl::queue& q) {
        if(values&&indices)return;
        values=sycl::malloc_device<float>(groups*capacity,q);
        indices=sycl::malloc_device<int32_t>(groups*capacity,q);
        if(!values||!indices){release(q);throw std::bad_alloc();}
    }
    void release(sycl::queue& q) {
        if(values)sycl::free(values,q);
        if(indices)sycl::free(indices,q);
        values=nullptr;indices=nullptr;
    }
};
inline sycl::event launch_sample_logits(sycl::queue& q,const float* logits,int n,
    int32_t* output,float* output_value,SamplingWorkspace& workspace,
    SamplingOptions options,uint64_t position,
    const std::vector<sycl::event>& deps={}) {
    options.validate();workspace.ensure(q);
    constexpr int G=SamplingWorkspace::groups,W=SamplingWorkspace::width;
    float* pv=workspace.values;int32_t* pi=workspace.indices;
    const int k=std::min(options.top_k,n);
    q.submit([&](sycl::handler& h){
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(G*W,W),[=](sycl::nd_item<1> it){
            const int g=it.get_group(0),lane=it.get_local_id(0);
            float previous=std::numeric_limits<float>::infinity();int previous_id=-1;
            for(int j=0;j<k;++j){
                float best=-std::numeric_limits<float>::infinity();int id=INT32_MAX;
                for(int i=g*W+lane;i<n;i+=G*W){
                    const float x=logits[i];
                    const bool after=(x<previous)||(x==previous&&i>previous_id);
                    if(after&&(x>best||(x==best&&i<id))){best=x;id=i;}
                }
                const float value=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<float>());
                const int index=sycl::reduce_over_group(it.get_group(),best==value?id:INT32_MAX,sycl::minimum<int>());
                if(lane==0){pv[j*G+g]=value;pi[j*G+g]=index;}
                previous=value;previous_id=index;
            }
        });
    });
    return q.submit([&](sycl::handler& h){
        sycl::local_accessor<float,1> scores(SamplingWorkspace::capacity,h);
        sycl::local_accessor<int32_t,1> ids(SamplingWorkspace::capacity,h);
        h.parallel_for(sycl::nd_range<1>(W,W),[=](sycl::nd_item<1> it){
            const int lane=it.get_local_id(0);
            float previous=std::numeric_limits<float>::infinity();int previous_id=-1;
            for(int j=0;j<k;++j){
                float best=-std::numeric_limits<float>::infinity();int id=INT32_MAX;
                for(int i=lane;i<G*k;i+=W){
                    const float x=pv[i];const int idx=pi[i];
                    if(idx==INT32_MAX)continue;
                    const bool after=x<previous||(x==previous&&idx>previous_id);
                    if(after&&(x>best||(x==best&&idx<id))){best=x;id=idx;}
                }
                const float value=sycl::reduce_over_group(it.get_group(),best,sycl::maximum<float>());
                const int index=sycl::reduce_over_group(it.get_group(),best==value?id:INT32_MAX,sycl::minimum<int>());
                if(lane==0){scores[j]=value;ids[j]=index;}
                previous=value;previous_id=index;
            }
            sycl::group_barrier(it.get_group());
            if(lane==0){
                const float maximum=scores[0];float total=0;
                for(int j=0;j<k;++j){
                    const float x=scores[j];
                    const float w=maximum==std::numeric_limits<float>::infinity()?(x==maximum?1.f:0.f):
                        sycl::exp((x-maximum)/options.temperature);
                    scores[j]=w;total+=w;
                }
                if(!(total>0)){*output=-1;return;}
                float mass=0;int keep=0;
                do{mass+=scores[keep++];}while(keep<k&&mass<options.top_p*total);
                const float draw=sampling_uniform_f32(options.seed,position)*mass;
                float cumulative=0;int chosen=keep-1;
                for(int j=0;j<keep;++j){cumulative+=scores[j];if(draw<cumulative){chosen=j;break;}}
                *output=ids[chosen];
                if(output_value)*output_value=logits[ids[chosen]];
            }
        });
    });
}
}
