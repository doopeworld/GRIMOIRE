// Reconstruction error of GRIMOIRE's INT4 round-to-nearest on synthetic weights (rows of 2560, a Student-t
// body with a few outliers, as LLM weight rows look); env picks the variant (read once per process).
#include "b70/weights.hpp"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>
namespace b70 { PackedWeight quantize(const float* src, int N, int K, Fmt fmt); }
int main(int argc, char** argv) {
    const int N = 512, K = 2560;
    std::mt19937 rng(42);
    std::student_t_distribution<float> t(4.0f);
    std::vector<float> w(size_t(N) * K);
    for (auto& x : w) x = 0.02f * t(rng);
    std::uniform_int_distribution<int> pos(0, K - 1);
    for (int n = 0; n < N; ++n) for (int j = 0; j < 3; ++j) w[size_t(n) * K + pos(rng)] *= 8.0f;  // outliers
    auto p = b70::quantize(w.data(), N, K, b70::Fmt::INT4);
    auto v = p.view();
    double e = 0, s = 0;
    for (int n = 0; n < N; ++n) for (int k = 0; k < K; ++k) {
        const double d = double(w[size_t(n) * K + k]) - double(v.at(n, k));
        e += d * d; s += double(w[size_t(n) * K + k]) * w[size_t(n) * K + k];
    }
    std::printf("%-28s group %d  relative MSE %.5f  (bytes/row %lld + %d scale/zero groups)\n",
                argc > 1 ? argv[1] : "", K / v.row_scales, e / s, (long long)v.row_bytes, v.row_scales);
}
