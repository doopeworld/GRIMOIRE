#include "b70/qwen35.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if(argc != 2) { std::fprintf(stderr,"usage: inspect-expert-zeros MODEL\n"); return 2; }
    b70::Qwen35Model m; std::string err;
    if(!m.load(argv[1],err)) { std::fprintf(stderr,"load: %s\n",err.c_str()); return 1; }
    std::array<unsigned long long,17> hist{};
    unsigned long long tensors=0, values=0;
    auto scan=[&](const b70::TensorRef& r) {
        if(!r.gptq) return true;
        std::vector<uint8_t> raw(size_t(r.qzeros_t.end-r.qzeros_t.begin));
        b70::TensorRef z=r; z.shard=r.qzeros_shard; z.t=r.qzeros_t;
        if(!m.read_raw(z,raw.data(),err)) return false;
        for(uint8_t x:raw) { ++hist[(x&15)+1]; ++hist[(x>>4)+1]; values+=2; }
        ++tensors; return true;
    };
    for(const auto& l:m.layers) for(size_t e=0;e<l.e_gate_p.size();++e)
        if(!scan(l.e_gate_p[e]) || !scan(l.e_up_p[e]) || !scan(l.e_down_p[e])) {
            std::fprintf(stderr,"read: %s\n",err.c_str()); return 1;
        }
    std::printf("tensors=%llu packed_zero_points=%llu\n",tensors,values);
    for(int z=1;z<=16;++z) if(hist[z])
        std::printf("zero=%d count=%llu %.6f%%\n",z,hist[z],100.0*hist[z]/values);
    return 0;
}
