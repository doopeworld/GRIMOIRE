// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// =====================================================================
//  imatrix.hpp  --  llama.cpp importance matrices for --proj int4
//
//  An importance matrix is the mean square of every linear layer's
//  input, per input channel, over a calibration run.  llama.cpp writes it
//  (llama-imatrix) and the GGUF publishers ship it beside their quants
//  (Unsloth: imatrix_unsloth.gguf; bartowski: *.imatrix).  GRIMOIRE reads
//  it to round a BF16 weight to INT4 by the error the layer's OUTPUT
//  sees, not the weight's own (quantize(..., importance)).  The INT4
//  format and every kernel are unchanged; only the codes differ.
//
//  Both file formats: the GGUF one (<tensor>.in_sum2 [K][n_mat] and
//  <tensor>.counts [n_mat], F32) and the older imatrix.dat.  Names are
//  llama.cpp's (blk.N.attn_q.weight); a checkpoint's HF names are mapped
//  (imatrix_gguf_name).  A tensor the file does not name, or names with a
//  different width, gets nullptr and keeps the plain rounding.
//  Implemented in src/quantize.cpp (linked by every binary that quantizes).
//  Host test: tools/test_imatrix.cpp.
// =====================================================================
#ifndef B70_IMATRIX_HPP
#define B70_IMATRIX_HPP

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace b70 {

class ImportanceMatrix {
public:
    // Reads the whole file (a few MB to a few hundred MB).  false + err on a
    // malformed file; an empty matrix is not an error.
    bool load(const std::string& path, std::string& err);
    // The K importances of a tensor (HF or GGUF name; an HF per-expert
    // tensor mlp.experts.E.* gets expert E's row of the merged entry), or
    // nullptr.  The pointer stays valid while the matrix lives.  count=false
    // leaves hits / misses alone (a lookup that quantizes nothing).
    const float* find(const std::string& tensor_name, int K, bool count = true) const;
    size_t entries() const { return m_.size(); }
    // How many find() calls returned a vector -- the loader prints it, so a
    // file that named nothing (wrong model) is visible, not silent.
    mutable std::atomic<int> hits{0}, misses{0};

private:
    struct Entry { int K = 0, n_mat = 0; std::vector<float> v; };  // [n_mat][K] means
    std::unordered_map<std::string, Entry> m_;
};

// llama.cpp's name for a checkpoint tensor ("" when unknown).  GGUF names
// pass through.  *expert receives E for an HF per-expert tensor
// (mlp.experts.E.gate_proj.weight -> ffn_gate_exps.weight, E), else -1.
std::string imatrix_gguf_name(const std::string& name, int* expert);

} // namespace b70
#endif // B70_IMATRIX_HPP
