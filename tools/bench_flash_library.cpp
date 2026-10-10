// Measure the production prefill-attention library and check sampled outputs
// against an independent fp64 causal-attention reference, including ragged tails.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace b70;

static uint32_t mix(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b;
    return x ^ (x >> 16);
}
static float fp8(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e ? std::ldexp(1.0f + float(m) / 8, e - 7)
                      : std::ldexp(float(m) / 8, -6);
    return b & 128 ? -v : v;
}

int main(int argc, char** argv) {
    const int H = argc > 4 ? std::atoi(argv[4]) : 24;
    const int KH = argc > 5 ? std::atoi(argv[5]) : 4;
    const int D = 256, CAP = 135168;
    const int T = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int start = argc > 2 ? std::atoi(argv[2]) : 12288;
    const int rounds = argc > 3 ? std::atoi(argv[3]) : 5;
    if (T < 1 || start < 0 || T > CAP - start || rounds < 1 ||
        H < 7 || KH < 1 || H % KH != 0) return 2;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device=%s T=%d start=%d cap=%d esimd=%s\n",
        q.get_device().get_info<sycl::info::device::name>().c_str(), T, start, CAP,
        std::getenv("GRIMOIRE_FLASH_ESIMD") ? std::getenv("GRIMOIRE_FLASH_ESIMD") : "auto");
    std::vector<uint8_t> k(size_t(KH) * D * CAP), v(k.size());
    for (size_t i = 0; i < k.size(); ++i) {
        const auto a = mix(uint32_t(i) + 17), b = mix(uint32_t(i) + 193);
        k[i] = uint8_t(0x20 + a % 48) | uint8_t((a >> 24) & 128);
        v[i] = uint8_t(0x20 + b % 48) | uint8_t((b >> 24) & 128);
    }
    std::vector<float> queries(size_t(T) * H * D), result(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
        queries[i] = float(int(mix(uint32_t(i) + 991) & 65535) - 32768) / 32768;
    auto* dk = sycl::malloc_device<uint8_t>(k.size(), q);
    auto* dv = sycl::malloc_device<uint8_t>(v.size(), q);
    auto* dq = sycl::malloc_device<float>(queries.size(), q);
    auto* out = sycl::malloc_device<float>(queries.size(), q);
    if (!dk || !dv || !dq || !out) return 3;
    q.memcpy(dk, k.data(), k.size()); q.memcpy(dv, v.data(), v.size());
    q.memcpy(dq, queries.data(), queries.size() * sizeof(float)); q.wait_and_throw();
    auto run = [&] {
        launch_flash_prefill_fast(q, dq, dk, dv, out, T, start, H, KH, D, CAP, 1.0f / 16, {});
    };
    run(); q.wait_and_throw();
    q.memcpy(result.data(), out, result.size() * sizeof(float)).wait();
    float lut[256]; for (int i = 0; i < 256; ++i) lut[i] = fp8(uint8_t(i));
    double max_error = 0, max_ref = 0;
    bool finite = true;
    for (float x : result) finite = finite && std::isfinite(x);
    for (int row : {0, T / 2, T - 1}) for (int head : {0, 1, 5, 6, H - 1}) {
        const int kh = head / (H / KH), length = start + row + 1;
        const float* qr = queries.data() + (size_t(row) * H + head) * D;
        std::vector<double> scores(size_t(length), 0), ref(D, 0);
        double maximum = -1e300, sum = 0;
        for (int s = 0; s < length; ++s) {
            double score = 0;
            for (int d = 0; d < D; ++d)
                score += double(qr[d]) * lut[k[(size_t(kh) * D + d) * CAP + s]];
            scores[size_t(s)] = score / 16;
            maximum = std::max(maximum, score / 16);
        }
        for (double& score : scores) { score = std::exp(score - maximum); sum += score; }
        for (int s = 0; s < length; ++s) {
            const double probability = scores[size_t(s)] / sum;
            const auto offset = (size_t(kh) * CAP + s) * D;
            for (int d = 0; d < D; ++d) ref[size_t(d)] += probability * lut[v[offset + d]];
        }
        const auto offset = (size_t(row) * H + head) * D;
        for (int d = 0; d < D; ++d) {
            max_ref = std::max(max_ref, std::fabs(ref[size_t(d)]));
            max_error = std::max(max_error, std::fabs(result[offset + d] - ref[size_t(d)]));
        }
    }
    uint64_t output_hash = 1469598103934665603ull;
    for (float value : result) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        output_hash = (output_hash ^ bits) * 1099511628211ull;
    }
    std::printf("output_hash=%016llx\n", static_cast<unsigned long long>(output_hash));
    const double relative = max_error / std::max(max_ref, 1e-12);
    if (!finite || relative > 0.02) {
        std::printf("oracle FAIL finite=%d error=%.6g ref=%.6g relative=%.6g\n",
                    finite, max_error, max_ref, relative);
        q.wait_and_throw(); return 1;
    }
    run(); q.wait_and_throw();
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < rounds; ++i) run();
    q.wait_and_throw();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count() / rounds;
    std::printf("oracle PASS sampled-relative=%.6g all-finite=1 kernel_ms=%.3f rounds=%d\n",
                relative, ms, rounds);
    sycl::free(dk, q); sycl::free(dv, q); sycl::free(dq, q); sycl::free(out, q);
    return 0;
}
