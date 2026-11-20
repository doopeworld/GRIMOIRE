// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace b70 {

struct QfnSpan {
    std::int64_t begin;
    std::int64_t count;
    constexpr std::int64_t end() const { return begin + count; }
};

enum class QfnLayerKind { unknown, gdn, attn, sparse_attn };
enum class QfnSharedLayout { unsupported, bf16, g128 };

// Resolved geometry of the dense target, not the draft. The layer kinds must
// reflect effective attention mode (a configured QSA layer is sparse_attn).
struct QfnShardShape {
    static constexpr int target_layers = 48;
    int layers = 0;
    std::array<QfnLayerKind, target_layers> layer_kinds{};
    int hidden = 0;
    int q_heads = 0;
    int kv_heads = 0;
    int attn_head_dim = 0;
    int q_proj_rows = 0;
    int gdn_k_heads = 0;
    int gdn_v_heads = 0;
    int gdn_k_head_dim = 0;
    int gdn_v_head_dim = 0;
    int gdn_conv_kernel = 0;
    int routed_experts = 0;
    int experts_per_token = 0;
    int routed_width = 0;
    int shared_width = 0;
};

// TP4/EP4 geometry only (qfn-enhanced/proposition.md §3.3). Spans are
// half-open coordinates in the unsharded tensor, not byte offsets.
class QfnShardPlan {
public:
    static constexpr int world = 4;
    static constexpr int no_local_expert = -1;

    // The caller validates tensor layouts on all applicable layers and selects
    // the shared layout from the shared tensors, not the output head format.
    // Refusal points to a static diagnostic, or is null on success.
    static std::optional<QfnShardPlan> make(const QfnShardShape& shape, int world_size,
                                            int rank, QfnSharedLayout layout,
                                            const char*& refusal);

    const int rank;
    const int layers;
    const int hidden;
    const int attn_head_dim;
    const int gdn_k_head_dim;
    const int gdn_v_head_dim;
    const int gdn_conv_kernel;
    const QfnSpan q_heads;     // Also the logical gate head span.
    const QfnSpan kv_heads;    // One whole logical head, replicated on the adjacent rank.
    const QfnSpan kv_proj_rows;
    const QfnSpan o_proj_k;
    const QfnSpan q_gate_rows; // Packed as [q0, gate0, q1, gate1, ...].
    const QfnSpan gdn_k_heads;
    const QfnSpan gdn_v_heads;
    const QfnSpan gdn_v_channels; // Gate (in_proj_z) rows and out_proj K columns.
    const std::array<QfnSpan, 3> gdn_qkv_rows; // [Q | K | V]; also conv channels.
    const std::array<QfnSpan, 2> gdn_ab_rows;  // [A | B], one row per V head in each.
    const QfnSpan routed_experts;
    const QfnSpan shared_channels; // Shared down_proj K columns.
    const std::array<QfnSpan, 2> shared_gate_up_rows; // [gate | up] in fused projection.

    constexpr int local_expert(std::int64_t global_id) const {
        return global_id < routed_experts.begin || global_id >= routed_experts.end()
            ? no_local_expert : static_cast<int>(global_id - routed_experts.begin);
    }

private:
    QfnShardPlan(const QfnShardShape& s, int r, QfnSharedLayout layout)
        : rank(r), layers(s.layers), hidden(s.hidden), attn_head_dim(s.attn_head_dim),
          gdn_k_head_dim(s.gdn_k_head_dim), gdn_v_head_dim(s.gdn_v_head_dim),
          gdn_conv_kernel(s.gdn_conv_kernel),
          q_heads{r * (s.q_heads / world), s.q_heads / world},
          kv_heads{r / (world / s.kv_heads), 1},
          kv_proj_rows{kv_heads.begin * attn_head_dim, attn_head_dim},
          o_proj_k{q_heads.begin * attn_head_dim, q_heads.count * attn_head_dim},
          q_gate_rows{2 * o_proj_k.begin, 2 * o_proj_k.count},
          gdn_k_heads{r * (s.gdn_k_heads / world), s.gdn_k_heads / world},
          gdn_v_heads{r * (s.gdn_v_heads / world), s.gdn_v_heads / world},
          gdn_v_channels{gdn_v_heads.begin * gdn_v_head_dim, gdn_v_heads.count * gdn_v_head_dim},
          gdn_qkv_rows{{{gdn_k_heads.begin * gdn_k_head_dim, gdn_k_heads.count * gdn_k_head_dim},
                        {(s.gdn_k_heads + gdn_k_heads.begin) * gdn_k_head_dim,
                         gdn_k_heads.count * gdn_k_head_dim},
                        {2LL * s.gdn_k_heads * gdn_k_head_dim + gdn_v_channels.begin,
                         gdn_v_channels.count}}},
          gdn_ab_rows{{gdn_v_heads, {s.gdn_v_heads + gdn_v_heads.begin, gdn_v_heads.count}}},
          routed_experts{r * (s.routed_experts / world), s.routed_experts / world},
          // Five whole groups over four ranks: rank 0 gets the one extra group.
          shared_channels(layout == QfnSharedLayout::g128
              ? QfnSpan{r == 0 ? 0 : (r + 1) * 128, r == 0 ? 256 : 128}
              : QfnSpan{r * (s.shared_width / world), s.shared_width / world}),
          shared_gate_up_rows{{shared_channels,
                               {s.shared_width + shared_channels.begin, shared_channels.count}}} {}
};

inline std::optional<QfnShardPlan> QfnShardPlan::make(const QfnShardShape& s, int world_size,
                                                       int rank, QfnSharedLayout layout,
                                                       const char*& refusal) {
    refusal = nullptr;
    if (world_size != world) refusal = "QFN shard plan requires TP4/EP4";
    else if (rank < 0 || rank >= world) refusal = "QFN rank is outside [0, 4)";
    else if (s.layers != QfnShardShape::target_layers || s.hidden != 2560)
        refusal = "QFN target requires 48 layers and hidden width 2560";
    else if (s.q_heads != 24 || s.kv_heads != 2 || s.attn_head_dim != 256 || s.q_proj_rows != 12288)
        refusal = "QFN attention requires 24 gated Q heads and 2 KV heads of width 256";
    else if (s.gdn_k_heads != 16 || s.gdn_v_heads != 48 || s.gdn_k_head_dim != 128 ||
             s.gdn_v_head_dim != 128 || s.gdn_conv_kernel != 4)
        refusal = "QFN GDN requires 16 K and 48 V heads of width 128 and conv kernel 4";
    else if (s.routed_experts != 512 || s.experts_per_token != 10 || s.routed_width != 640)
        refusal = "QFN routing requires top-10 of 512 whole experts of width 640";
    else if (s.shared_width != 640 || (layout != QfnSharedLayout::bf16 && layout != QfnSharedLayout::g128))
        refusal = "QFN shared expert requires width 640 and BF16 or g128 layout";
    if (refusal) return std::nullopt;

    for (std::size_t i = 0; i < s.layer_kinds.size(); ++i) {
        const auto expected = i % 4 == 3 ? QfnLayerKind::attn : QfnLayerKind::gdn;
        if (s.layer_kinds[i] != expected) {
            refusal = "QFN dense target requires 12 repeats of [GDN, GDN, GDN, attention]";
            return std::nullopt;
        }
    }
    return QfnShardPlan(s, rank, layout);
}

} // namespace b70
