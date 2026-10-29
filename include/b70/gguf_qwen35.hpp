// GRIMOIRE -- qwen35/qwen35moe GGUF source adapter. SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "qwen35.hpp"
#include <string>

namespace b70 {
// Synthesizes the same text_config fields consumed by Qwen35Model::load.
// If sidecar_json is nonempty, every derived field is checked against it.
bool gguf_qwen35_config(const GGUFFile& file, const std::string& sidecar_json,
                         std::string& config_json, std::string& err);
bool gguf_qwen35_index(Qwen35Model& model, std::string& err);
bool gguf_qwen3_config(const GGUFFile& file, const std::string& sidecar_json,
                       std::string& config_json, std::string& err);
bool gguf_qwen3_index(Qwen35Model& model, std::string& err);
bool gguf_gemma4_config(const GGUFFile& file, const std::string& sidecar_json,
                        std::string& config_json, std::string& err);
bool gguf_gemma4_index(Qwen35Model& model, std::string& err);
// V head permutation used by the inverse adapter. The forward direction is
// implemented independently in tests/test_gguf_qwen35.cpp.
uint64_t gguf_v_tiled_index(uint64_t grouped, uint64_t nk, uint64_t nv);
// Source grouped head for a destination tiled head. Shared by the host
// permutation test and the ssm_out device activation gather.
inline uint64_t gguf_v_grouped_index(uint64_t tiled, uint64_t nk, uint64_t nv) {
    return (tiled % nk) * (nv / nk) + tiled / nk;
}
template <class T>
inline void gguf_gather_v_input(const T* grouped, T* tiled, uint64_t rows,
                                uint64_t nk, uint64_t nv, uint64_t width) {
    for(uint64_t row=0;row<rows;++row)
        for(uint64_t head=0;head<nv;++head)
            for(uint64_t i=0;i<width;++i)
                tiled[row*nv*width+head*width+i]=
                    grouped[row*nv*width+gguf_v_grouped_index(head,nk,nv)*width+i];
}
// Validate recovery of a BF16 A_log from GGUF's float32 -exp(A_log).
bool gguf_a_log_recovery_ok(float a, float recovered);
}
