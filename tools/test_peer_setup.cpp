// SPDX-License-Identifier: GPL-3.0-or-later
// Startup negotiation on one GPU: disabled/auto fallback and strict refusal.
#include "b70/peer_copy.hpp"
#include <sys/wait.h>
#include <cstdio>
int main(int argc,char** argv) {
    try {
        if(argc==4) {
            const int fd=std::atoi(argv[1]),rank=std::atoi(argv[2]);
            const bool expected=std::atoi(argv[3]);
            sycl::queue q{sycl::gpu_selector_v,sycl::property::queue::in_order{}};
            b70::PeerCopy peer;std::string why;
            const bool ok=peer.setup(q,fd,rank==0,why);
            if(rank==0)std::printf("setup mode=%s accepted=%d active=%d reason=%s\n",std::getenv("GRIMOIRE_P2P"),ok,peer.active(),why.c_str());
            const bool pass=ok==expected && !peer.active();
            ::close(fd);return pass?0:2;
        }
        struct Case{const char* mode;bool expected;};
        for(auto c:{Case{"0",true},Case{"auto",true},Case{"required",false},Case{"invalid",false}}) {
            ::setenv("GRIMOIRE_P2P",c.mode,1);
            int s[2];if(::socketpair(AF_UNIX,SOCK_STREAM,0,s))return 3;
            pid_t kids[2];
            for(int r=0;r<2;++r) {
                kids[r]=::fork();if(kids[r]<0)return 4;
                if(!kids[r]) {
                    ::close(s[1-r]);auto fd=std::to_string(s[r]),rank=std::to_string(r);
                    ::execl("/proc/self/exe",argv[0],fd.c_str(),rank.c_str(),c.expected?"1":"0",nullptr);::_exit(5);
                }
            }
            ::close(s[0]);::close(s[1]);bool pass=true;
            for(auto kid:kids){int st=0;if(::waitpid(kid,&st,0)<0 || !WIFEXITED(st) || WEXITSTATUS(st))pass=false;}
            if(!pass)return 6;
        }
        std::puts("PEER_SETUP_PASS");return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"peer setup test: %s\n",e.what());return 7;}
}
