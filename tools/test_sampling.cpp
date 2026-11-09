// SPDX-License-Identifier: GPL-3.0-or-later
#include "sample_logits.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
#include <chrono>
int main(){
    using namespace b70;
    sycl::queue q{sycl::property::queue::in_order{}};
    std::cout<<q.get_device().get_info<sycl::info::device::name>()<<std::endl;
    constexpr int N=248320,R=8192;
    auto* logits=sycl::malloc_shared<float>(N,q);
    auto* output=sycl::malloc_shared<int32_t>(R,q);
    auto* value=sycl::malloc_shared<float>(1,q);
    assert(logits&&output&&value);
    std::fill(logits,logits+N,-100.f);
    // Known distribution 0.6, 0.3, 0.1 after temperature, independent of
    // vocabulary size and candidate location (including partition boundaries).
    const int ids[]={2,131071,248319};
    for(int j=0;j<3;++j)logits[ids[j]]=.7f*std::log(float(j==0?.6:j==1?.3:.1));
    SamplingOptions o;o.temperature=.7f;o.top_p=1;o.top_k=3;o.seed=123;o.seeded=true;
    SamplingWorkspace w;
    for(int i=0;i<R;++i)launch_sample_logits(q,logits,N,output+i,value,w,o,i);
    q.wait_and_throw();
    int count[3]={};
    for(int i=0;i<R;++i){int found=-1;for(int j=0;j<3;++j)if(output[i]==ids[j])found=j;assert(found>=0);++count[found];}
    assert(std::abs(double(count[0])/R-.6)<.02);
    assert(std::abs(double(count[1])/R-.3)<.02);
    assert(std::abs(double(count[2])/R-.1)<.02);
    const int repeat=output[83];
    launch_sample_logits(q,logits,N,output,value,w,o,83).wait();assert(output[0]==repeat);
    // top_p .7 keeps the .6 and .3 tokens and renormalizes them to 2/3,1/3.
    o.top_p=.7f;
    for(int i=0;i<2048;++i)launch_sample_logits(q,logits,N,output+i,value,w,o,i);
    q.wait_and_throw();int a=0;
    for(int i=0;i<2048;++i){assert(output[i]==ids[0]||output[i]==ids[1]);a+=output[i]==ids[0];}
    assert(std::abs(double(a)/2048-2./3)<.035);
    o.top_k=1;launch_sample_logits(q,logits,N,output,value,w,o,1).wait();assert(output[0]==ids[0]);
    // Equal logits use the lowest token index to decide the top-k set.
    std::fill(logits,logits+N,0.f);o.top_k=2;o.top_p=1;
    for(int i=0;i<32;++i)launch_sample_logits(q,logits,N,output+i,value,w,o,i);
    q.wait_and_throw();for(int i=0;i<32;++i)assert(output[i]>=0&&output[i]<2);
    // A vocabulary smaller than k still forms a normalized distribution.
    o.top_k=64;launch_sample_logits(q,logits,3,output,value,w,o,2).wait();assert(output[0]>=0&&output[0]<3);
    w.release(q);sycl::free(logits,q);sycl::free(output,q);sycl::free(value,q);
    std::cout<<"SAMPLING_DISTRIBUTION_PASS "<<count[0]<<" "<<count[1]<<" "<<count[2]<<std::endl;
}
