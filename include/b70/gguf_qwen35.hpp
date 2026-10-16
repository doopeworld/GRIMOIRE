// GRIMOIRE -- qwen35moe GGUF source adapter. SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "qwen35.hpp"
#include <string>

namespace b70 {
// Synthesizes the same text_config fields consumed by Qwen35Model::load.
// If sidecar_json is nonempty, every derived field is checked against it.
bool gguf_qwen35_config(const GGUFFile& file, const std::string& sidecar_json,
                         std::string& config_json, std::string& err);
bool gguf_qwen35_index(Qwen35Model& model, std::string& err);
// V head permutation used by the inverse adapter. The forward direction is
// implemented independently in tests/test_gguf_qwen35.cpp.
uint64_t gguf_v_tiled_index(uint64_t grouped, uint64_t nk, uint64_t nv);
// Validate recovery of a BF16 A_log from GGUF's float32 -exp(A_log).
bool gguf_a_log_recovery_ok(float a, float recovered);
}
