#include "b70/b70q4.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace b70 {
namespace {
int round_up(int x, int a) { return (x + a - 1) / a * a; }
}

size_t B70Q4Weight::payload_offset(int n, int k) const {
    const int nt = n / kB70Q4N, ni = n % kB70Q4N;
    const int kt = k / kB70Q4K, ki = k % kB70Q4K;
    const int ktiles = padded_k / kB70Q4K;
    const size_t tile = size_t(nt * ktiles + kt) * kB70Q4N * kB70Q4K / 2;
    return tile + size_t(ki / 4) * kB70Q4N * 2 + size_t(ni) * 2
                + size_t((ki & 2) != 0);
}

size_t B70Q4Weight::scale_offset(int n, int k) const {
    const int nt = n / kB70Q4N, ni = n % kB70Q4N;
    const int kt = k / kB70Q4K;
    const int ktiles = padded_k / kB70Q4K;
    return size_t(nt * ktiles + kt) * kB70Q4N + ni;
}

int8_t B70Q4Weight::code(int n, int k) const {
    const uint8_t byte = payload[payload_offset(n, k)];
    const uint8_t nib = (k & 1) ? uint8_t(byte >> 4) : uint8_t(byte & 15);
    return int8_t((nib & 8) ? int(nib) - 16 : int(nib));
}

float B70Q4Weight::at(int n, int k) const {
    return float(code(n, k)) * bf16_to_f32(scales[scale_offset(n, k)]);
}

B70Q4Weight quantize_b70q4(const float* src, int N, int K) {
    if (!src || N <= 0 || K <= 0) throw std::invalid_argument("invalid B70Q4 matrix");
    B70Q4Weight w;
    w.N = N; w.K = K;
    w.padded_n = round_up(N, kB70Q4N);
    w.padded_k = round_up(K, kB70Q4K);
    w.payload.assign(size_t(w.padded_n) * w.padded_k / 2, 0);
    w.scales.assign(size_t(w.padded_n) * (w.padded_k / kB70Q4K), f32_to_bf16(1.0f));

    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < w.padded_k; kb += kB70Q4K) {
            const int count = std::max(0, std::min(kB70Q4K, K - kb));
            float amax = 0.0f;
            for (int i = 0; i < count; ++i)
                amax = std::max(amax, std::fabs(src[size_t(n) * K + kb + i]));
            float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
            const bf16_t sb = f32_to_bf16(scale);
            scale = bf16_to_f32(sb);
            w.scales[w.scale_offset(n, kb)] = sb;
            for (int i = 0; i < count; ++i) {
                int q = int(std::nearbyint(src[size_t(n) * K + kb + i] / scale));
                q = std::max(-7, std::min(7, q));
                const int k = kb + i;
                uint8_t& byte = w.payload[w.payload_offset(n, k)];
                const uint8_t nib = uint8_t(q) & 15;
                if (k & 1) byte = uint8_t((byte & 0x0f) | (nib << 4));
                else       byte = uint8_t((byte & 0xf0) | nib);
            }
        }
    }
    return w;
}

} // namespace b70
