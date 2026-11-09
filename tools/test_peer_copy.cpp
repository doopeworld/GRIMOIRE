// SPDX-License-Identifier: GPL-3.0-or-later
// Re-exec workers: never fork a live SYCL runtime. Same-device IPC validates
// protocol/numerics only; it is not evidence of physical peer bandwidth.
#include "b70/peer_copy.hpp"
#include <sys/wait.h>
#include <cstdio>
#include <vector>
#include <cmath>

static int worker(int fd,int rank) {
    sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
    b70::PeerCopy peer; std::string why;
    if(!peer.setup(q,fd,rank==0,why,true) || !peer.active()) {
        std::fprintf(stderr,"rank %d setup: %s\n",rank,why.c_str());return 2;
    }
    constexpr size_t n=600013; // multiple chunks, including a short tail
    std::vector<float> input(n),result(n);
    float* data=sycl::malloc_device<float>(n,q);
    float* extra=sycl::malloc_host<float>(n,q);
    if(!data || !extra)return 3;
    for(size_t i=0;i<n;++i) {input[i]=float(int(i%257)-128)/32+rank;extra[i]=float(rank+1)/8;}
    for(bool bf16:{false,true}) {
        q.memcpy(data,input.data(),n*sizeof(float)).wait_and_throw();
        if(!peer.allreduce(q,data,n,rank,bf16,extra))return 4;
        q.memcpy(result.data(),data,n*sizeof(float)).wait_and_throw();
        // These dyadic values are exactly representable in both formats.
        for(size_t i=0;i<n;++i) {
            float expected=2*float(int(i%257)-128)/32+1+3.0f/8;
            if(result[i]!=expected){std::fprintf(stderr,"sum mismatch %zu\n",i);return 5;}
        }
    }
    if(rank==0) {
        q.memcpy(data,input.data(),n*sizeof(float)).wait_and_throw();
        if(!peer.send(q,data,n) || !peer.recv(q,data,n))return 6;
        q.memcpy(result.data(),data,n*sizeof(float)).wait_and_throw();
        for(size_t i=0;i<n;++i)if(result[i]!=input[i]+2)return 7;
    } else {
        if(!peer.recv(q,data,n))return 8;
        q.parallel_for(sycl::range<1>(n),[=](sycl::id<1> i){data[i]+=2;}).wait_and_throw();
        if(!peer.send(q,data,n))return 9;
    }
    q.wait_and_throw();peer.reset();sycl::free(data,q);sycl::free(extra,q);::close(fd);
    return 0;
}
int main(int argc,char** argv) {
    try {
        if(argc==3)return worker(std::atoi(argv[1]),std::atoi(argv[2]));
        ::setenv("GRIMOIRE_P2P","required",1);
        ::setenv("GRIMOIRE_P2P_CHUNK_MB","1",1);
        int fds[2];if(::socketpair(AF_UNIX,SOCK_STREAM,0,fds))return 10;
        pid_t kids[2];
        for(int r=0;r<2;++r) {
            kids[r]=::fork();if(kids[r]<0)return 11;
            if(!kids[r]) {
                ::close(fds[1-r]);std::string fd=std::to_string(fds[r]),rank=std::to_string(r);
                ::execl("/proc/self/exe",argv[0],fd.c_str(),rank.c_str(),nullptr);::_exit(12);
            }
        }
        ::close(fds[0]);::close(fds[1]);bool ok=true;
        for(auto k:kids){int status=0;if(::waitpid(k,&status,0)<0 || !WIFEXITED(status) || WEXITSTATUS(status))ok=false;}
        std::puts(ok?"PEER_COPY_SAME_GPU_PASS (protocol only)":"PEER_COPY_FAIL");return ok?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"peer test: %s\n",e.what());return 13;}
}
