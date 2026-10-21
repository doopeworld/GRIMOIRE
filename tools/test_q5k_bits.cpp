// B70 root-cause probe: summing (byte >> lane) & 1 over 32 qh bytes inside a
// divergent `if (lane < 8)` region (the old Q5_K lane code) versus variants.
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdint>

template <int V>
static int run(sycl::queue& q, const uint8_t* p, int* out) {
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int lane = it.get_local_id(0);
        int r = 0;
        if (V == 0) {           // old form: divergent guard, two 16-byte runs
            if (lane < 8) {
                for (int i = 0; i < 16; ++i) r += (p[16 + i] >> lane) & 1;
                for (int i = 0; i < 16; ++i) r += (p[32 + i] >> lane) & 1;
            }
        } else if (V == 1) {    // no guard (all 16 lanes shift; lanes >= 8 result ignored)
            for (int i = 0; i < 16; ++i) r += (p[16 + i] >> lane) & 1;
            for (int i = 0; i < 16; ++i) r += (p[32 + i] >> lane) & 1;
        } else if (V == 2) {    // guard, but one single 32-iteration loop
            if (lane < 8)
                for (int i = 0; i < 32; ++i) r += (p[16 + i] >> lane) & 1;
        } else if (V == 3) {    // guard, two loops, no unrolling
            if (lane < 8) {
                #pragma unroll 1
                for (int i = 0; i < 16; ++i) r += (p[16 + i] >> lane) & 1;
                #pragma unroll 1
                for (int i = 0; i < 16; ++i) r += (p[32 + i] >> lane) & 1;
            }
        } else if (V == 5) {    // shift by lane >> 1 (new Q5_K mapping), all lanes active
            for (int i = 0; i < 32; ++i) r += (p[16 + i] >> (lane >> 1)) & 1;
        } else if (V == 6) {    // shift by lane & 7, all lanes active
            for (int i = 0; i < 32; ++i) r += (p[16 + i] >> (lane & 7)) & 1;
        } else if (V == 7) {    // shift by lane, but the byte is first widened and masked to 8 bits via 64-bit pair (production load16 style)
            uint64_t a, b; __builtin_memcpy(&a, p + 16, 8); __builtin_memcpy(&b, p + 24, 8);
            uint64_t c, d; __builtin_memcpy(&c, p + 32, 8); __builtin_memcpy(&d, p + 40, 8);
            if (lane < 8) {
                for (int i = 0; i < 16; ++i) r += int(((i < 8 ? a >> (i * 8) : b >> ((i - 8) * 8)) & 255u) >> lane) & 1;
                for (int i = 0; i < 16; ++i) r += int(((i < 8 ? c >> (i * 8) : d >> ((i - 8) * 8)) & 255u) >> lane) & 1;
            }
        } else if (V == 4) {    // guard, per-iteration values through an opaque store (checks the loads)
            if (lane < 8) {
                int t[32];
                for (int i = 0; i < 32; ++i) t[i] = (p[16 + i] >> lane) & 1;
                for (int i = 0; i < 32; ++i) r += t[i];
            }
        }
        out[lane] = r;
    }).wait_and_throw();
    int bad = 0;
    for (int lane = 0; lane < 8; ++lane) {
        int ref = 0;
        const int sh = V == 5 ? lane >> 1 : V == 6 ? lane & 7 : lane;
        for (int i = 0; i < 32; ++i) ref += (p[16 + i] >> sh) & 1;
        if (out[lane] != ref) { ++bad; std::printf("  V=%d lane=%d dev=%d ref=%d\n", V, lane, out[lane], ref); }
    }
    std::printf("variant %d mismatching lanes=%d\n", V, bad);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    auto* p = sycl::malloc_shared<uint8_t>(256, q);
    auto* out = sycl::malloc_shared<int>(16, q);
    for (int i = 0; i < 256; ++i) p[i] = uint8_t((i * 73 + 5) & 255);
    int bad = run<0>(q, p, out) + run<1>(q, p, out) + run<2>(q, p, out) + run<3>(q, p, out) + run<4>(q, p, out) + run<5>(q, p, out) + run<6>(q, p, out) + run<7>(q, p, out);
    return bad != 0;
}
