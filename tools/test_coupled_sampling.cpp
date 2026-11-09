// SPDX-License-Identifier: GPL-3.0-or-later
#include "sample_logits.hpp"
#include <cassert>
#include <iostream>
int main(){
    using namespace b70;
    sycl::queue q{sycl::property::queue::in_order{}};
    constexpr int N=8192;
    auto* p=sycl::malloc_shared<float>(3,q);auto* ql=sycl::malloc_shared<float>(3,q);
    auto* output=sycl::malloc_shared<int32_t>(N,q);auto* accepted=sycl::malloc_shared<int32_t>(N,q);
    auto* proposal=sycl::malloc_device<int32_t>(1,q);
    auto* stored=sycl::malloc_device<SparseSamplingDistribution>(1,q);
    assert(p&&ql&&output&&accepted&&proposal&&stored);
    p[0]=std::log(.6f)*.7f;p[1]=std::log(.3f)*.7f;p[2]=std::log(.1f)*.7f;
    ql[0]=std::log(.5f)*.7f;ql[1]=std::log(.2f)*.7f;ql[2]=std::log(.3f)*.7f;
    SamplingOptions options;options.temperature=.7f;options.top_p=1;options.top_k=3;options.seed=123;
    SamplingWorkspace work;
    for(int i=0;i<N;++i){
        auto draft_options=options;draft_options.seed^=kProposalStream;
        launch_sample_logits(q,ql,3,proposal,nullptr,work,draft_options,i,{},stored,nullptr,9);
        launch_sample_logits(q,p,3,output+i,nullptr,work,options,i,{},nullptr,stored,9,accepted+i);
    }
    q.wait_and_throw();int count[3]={},acc=0;
    for(int i=0;i<N;++i){assert(output[i]>=0&&output[i]<3);++count[output[i]];acc+=accepted[i];}
    assert(std::abs(double(count[0])/N-.6)<.025);
    assert(std::abs(double(count[1])/N-.3)<.025);
    assert(std::abs(double(count[2])/N-.1)<.025);
    assert(std::abs(double(acc)/N-.8)<.025);
    // A record from another epoch or seed must be ignored, even if the
    // physical ring slot and absolute position match the current request.
    for(int i=0;i<64;++i){
        launch_sample_logits(q,p,3,output+i,nullptr,work,options,N-1,{},nullptr,stored,10,accepted+i);
    }
    q.wait_and_throw();for(int i=0;i<64;++i)assert(accepted[i]==0&&output[i]==output[0]);
    launch_sample_logits(q,p,3,output,nullptr,work,options,N,{},nullptr,stored,9,accepted).wait();
    assert(accepted[0]==0);
    auto different_seed=options;different_seed.seed=124;
    launch_sample_logits(q,p,3,output,nullptr,work,different_seed,N-1,{},nullptr,stored,9,accepted).wait();
    assert(accepted[0]==0);
    work.release(q);
    sycl::free(p,q);sycl::free(ql,q);sycl::free(output,q);sycl::free(accepted,q);sycl::free(proposal,q);sycl::free(stored,q);
    std::cout<<"COUPLED_SAMPLING_GPU_PASS "<<count[0]<<" "<<count[1]<<" "<<count[2]<<" accepted "<<acc<<"/"<<N<<std::endl;
}
