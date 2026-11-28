// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "kernels.hpp"
#include "one_hot.hpp"
#include "b70/moe.hpp"
#include <memory>

namespace qfn_xe2 {
#ifndef QFN_XE2_SINGLE_ONLY   // single-rank probe links no engine kernels
class DecodeM8 {
    struct Scratch;
    std::unique_ptr<Scratch> scratch;
public:
    explicit DecodeM8(sycl::queue& q);
    ~DecodeM8();
    void run(const b70::MoeLayer& moe, const int32_t* routes, const float* weights,
             const float* input, float* output, const float* reference_hidden,
             int rank, int pos, int layer);
};
#endif
void round_n64_gate_up(sycl::queue& q, const int32_t* live_rows, float* out);
void launch_signed_gate_up(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                           const int32_t* counts, const int32_t* live_rows,
                           b70::sycl_bf16* scratch, int32_t* atomic, float* out, bool fp32_output = false);
void launch_signed_down(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                        const int32_t* counts, int32_t* atomic, float* out);
void launch_signed_gate_up_p32(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                               const int32_t* counts, int32_t* atomic, float* out);
void launch_signed_down_p32(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                            const int32_t* counts, int32_t* atomic, float* out);
void launch_signed_gate_up_tail(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                                const int32_t* counts, int32_t* atomic, float* out);
void launch_signed_down_tail(sycl::queue& q, const b70::QuantWeight& w, const b70::sycl_bf16* x,
                             const int32_t* counts, int32_t* atomic, float* out);
bool compare_gate_up(sycl::queue& q, const b70::QuantWeight& w, int layer, int rank,
                     const b70::sycl_bf16* x, const int32_t* counts, const int32_t* offsets,
                     const b70::QfnMoeTile* tiles, const int32_t* live_tiles, int max_tiles,
                     std::span<const HotRow> oracle_rows = {}, bool fp32_output = false);
}
