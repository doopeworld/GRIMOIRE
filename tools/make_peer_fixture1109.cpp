// SPDX-License-Identifier: GPL-3.0-or-later
// Small transport fixture satisfying Megatron's 128-element shard boundaries.
// Random weights test execution/protocols; output is not model-quality evidence.
#include "mini_model.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
static void replace(std::string& s,const std::string& from,const std::string& to) {
    auto at=s.find(from);if(at==std::string::npos)throw std::runtime_error("fixture config mismatch");
    s.replace(at,from.size(),to);
}
int main(int argc,char** argv) {
    if(argc!=2)return 1;
    auto a=b70::mini::dense(4);
    replace(a.config,"\"hidden_size\": 64","\"hidden_size\": 256");
    replace(a.config,"\"head_dim\": 16","\"head_dim\": 64");
    replace(a.config,"\"intermediate_size\": 128","\"intermediate_size\": 512");
    for(auto& t:a.tensors) {
        const auto& n=t.name;
        if(t.shape.size()==1) {
            const int width=n.find("q_norm")!=std::string::npos || n.find("k_norm")!=std::string::npos?64:256;
            t.shape={width};t.fill.assign(size_t(width),1.0f);
        } else if(n.find("embed_tokens")!=std::string::npos || n=="lm_head.weight")t.shape={128,256};
        else if(n.find("q_proj")!=std::string::npos || n.find("o_proj")!=std::string::npos)t.shape={256,256};
        else if(n.find("k_proj")!=std::string::npos || n.find("v_proj")!=std::string::npos)t.shape={128,256};
        else if(n.find("gate_proj")!=std::string::npos || n.find("up_proj")!=std::string::npos)t.shape={512,256};
        else if(n.find("down_proj")!=std::string::npos)t.shape={256,512};
        else throw std::runtime_error("unexpected fixture tensor: "+n);
    }
    b70::mini::write_model(argv[1],a);
    std::array<int,256> codes{};int next=256;
    for(int i=0;i<256;++i)codes[size_t(i)]=(i>=33&&i<=126)||(i>=161&&i<=172)||i>=174?i:next++;
    std::ofstream out(std::filesystem::path(argv[1])/"tokenizer.json");
    out<<"{\"model\":{\"type\":\"BPE\",\"vocab\":{";
    for(int i=0;i<128;++i){char key[16];std::snprintf(key,sizeof key,"\\u%04x",codes[size_t(i)]);out<<(i?",":"")<<'"'<<key<<"\":"<<i;}
    out<<"},\"merges\":[]},\"added_tokens\":[]}";
    return out?0:2;
}
