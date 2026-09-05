// Which 4-bit format can actually be consumed at full bandwidth on B70?
// Identical memory traffic, identical access pattern (16 B per lane, sub-group
// contiguous, 4 streams in flight -- the pattern that measured 585 GB/s raw).
// The ONLY difference is the dequant math.
//
//   raw     : no dequant, the ceiling
//   int4_sc : (nibble - 8) * scale   -- shift/mask/sub/mul, pure ALU, no tables
//   mxfp4   : E2M1 nibble -> LUT, E8M0 exp -> LUT (what GRIMOIRE does today,
//             "two SLM gathers per weight byte")
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
using namespace sycl;
constexpr int SG = 16;
constexpr uint32_t PHI = 2654435761u;

int main(int argc, char** argv) {
    const size_t GB = size_t(argc > 1 ? atoi(argv[1]) : 2);
    const int it_n  = argc > 2 ? atoi(argv[2]) : 3;
    queue q{gpu_selector_v, property::queue::in_order()};
    std::printf("device : %s\n", q.get_device().get_info<info::device::name>().c_str());

    const size_t bytes = GB << 30, words = bytes / 4, chunks = words / 4;
    uint32_t* arena = malloc_device<uint32_t>(words, q);
    q.parallel_for(range<1>(words), [=](id<1> i){ arena[i] = PHI * uint32_t(i); }).wait();

    const size_t threads = 1ull << 20, groups = threads / SG;
    float* out = malloc_device<float>(threads, q);
    // separate scales array: one byte per 32 weights (per uint32 word)
    const size_t nsc = words;
    uint8_t* scales = malloc_device<uint8_t>(nsc, q);
    q.parallel_for(range<1>(nsc), [=](id<1> i){ scales[i] = uint8_t(125 + (i % 5)); }).wait();
    const size_t per_g = chunks / groups, step = SG * 4;

    auto run = [&](const char* name, int mode) {
        double best = 1e30;
        for (int r = 0; r < it_n + 1; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            q.submit([&](handler& h) {
                local_accessor<float,1> e2m1(16, h);     // E2M1 nibble -> float
                local_accessor<float,1> e8m0(256, h);    // E8M0 byte -> 2^(x-127)
                h.parallel_for(nd_range<1>(threads, 256), [=](nd_item<1> it)
                    [[sycl::reqd_sub_group_size(SG)]] {
                    const int lid = int(it.get_local_id(0));
                    if (lid < 16) e2m1[lid] = float(lid & 7) * ((lid & 8) ? -1.f : 1.f);
                    for (int i = lid; i < 256; i += 256)
                        e8m0[i] = sycl::exp2(float(i) - 127.f);
                    group_barrier(it.get_group());

                    const size_t gid = it.get_global_id(0);
                    const size_t g = gid / SG; const int lane = int(gid % SG);
                    const vec<uint32_t,4>* base =
                        reinterpret_cast<const vec<uint32_t,4>*>(arena) + g * per_g;
                    float a0=0,a1=0,a2=0,a3=0;
                    for (size_t i = 0; i + step <= per_g; i += step) {
                        vec<uint32_t,4> v[4] = { base[i+lane], base[i+SG+lane],
                                                 base[i+2*SG+lane], base[i+3*SG+lane] };
                        float* acc[4] = {&a0,&a1,&a2,&a3};
                        #pragma unroll
                        for (int s = 0; s < 4; ++s) {
                            float sum = 0.f;
                            if (mode == 0) {                       // raw
                                #pragma unroll
                                for (int w = 0; w < 4; ++w) sum += float(v[s][w] & 0xFF);
                            } else if (mode == 1) {                // int4 + scale, ALU
                                const float sc = 0.015625f;        // inline block scale
                                #pragma unroll
                                for (int w = 0; w < 4; ++w) {
                                    const uint32_t word = v[s][w];
                                    #pragma unroll
                                    for (int b = 0; b < 8; ++b)
                                        sum += (float(int((word >> (4*b)) & 0xF) - 8)) * sc;
                                }
                            } else if (mode == 3) {                // scales from a SEPARATE array
                                #pragma unroll
                                for (int w = 0; w < 4; ++w) {
                                    const uint32_t word = v[s][w];
                                    // one scale byte per 8 nibbles, from a
                                    // distant array indexed by chunk -- this is
                                    // what QuantWeight.scales actually does
                                    const size_t ci = (i + size_t(s)*SG + lane)*4 + w;
                                    const float scale = e8m0[scales[ci] & 0xFF];
                                    #pragma unroll
                                    for (int b = 0; b < 8; ++b)
                                        sum += e2m1[(word >> (4*b)) & 0xF] * scale;
                                }
                            } else {                               // mxfp4, two SLM gathers
                                #pragma unroll
                                for (int w = 0; w < 4; ++w) {
                                    const uint32_t word = v[s][w];
                                    const float scale = e8m0[(word >> 24) & 0xFF];
                                    #pragma unroll
                                    for (int b = 0; b < 8; ++b)
                                        sum += e2m1[(word >> (4*b)) & 0xF] * scale;
                                }
                            }
                            *acc[s] += sum;
                        }
                    }
                    out[gid] = a0+a1+a2+a3;
                });
            });
            q.wait();
            auto t1 = std::chrono::high_resolution_clock::now();
            double s = std::chrono::duration<double>(t1-t0).count();
            if (r) best = std::min(best, s);
        }
        std::printf("%-9s %7.1f GB/s\n", name, double(bytes)/best/1e9);
    };

    std::printf("arena  : %zu GB, same traffic + access pattern, only dequant differs\n\n", GB);
    run("raw",     0);
    run("int4_sc", 1);
    run("mxfp4",   2);
    run("mx+sep",  3);   // scales in a separate array
    std::printf("\nraw-load ceiling measured earlier: 585.5 GB/s. Bar: 550.\n");
    free(arena,q); free(out,q); free(scales,q);
    return 0;
}
