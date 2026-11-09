// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace b70 {
// Greedy remains the default. Nonzero temperature uses top-k then nucleus
// truncation, renormalized before drawing. The seed and absolute output
// position identify each draw, independently of batching or rejected drafts.
struct SamplingOptions {
    float temperature=0, top_p=1;
    int top_k=20;
    uint64_t seed=0;
    bool seeded=false;
    bool enabled() const { return temperature>0; }
    void validate() const {
        if(!std::isfinite(temperature)||temperature<0)
            throw std::invalid_argument("temperature must be finite and nonnegative");
        if(!std::isfinite(top_p)||top_p<=0||top_p>1)
            throw std::invalid_argument("top_p must be in (0,1]");
        if(enabled()&&(top_k<1||top_k>64))
            throw std::invalid_argument("sampled decoding supports top_k from 1 to 64");
    }
};
inline uint64_t sampling_mix(uint64_t v) {
    v=(v^(v>>30))*UINT64_C(0xbf58476d1ce4e5b9);
    v=(v^(v>>27))*UINT64_C(0x94d049bb133111eb);
    return v^(v>>31);
}
inline float sampling_uniform_f32(uint64_t seed,uint64_t position) {
    return (float(sampling_mix(seed+UINT64_C(0x9e3779b97f4a7c15)*(position+1))>>41)+0.5f)*0x1p-23f;
}
inline double sampling_uniform(uint64_t seed,uint64_t position) {
    return (double(sampling_mix(seed+UINT64_C(0x9e3779b97f4a7c15)*(position+1))>>11)+0.5)*0x1p-53;
}
}
