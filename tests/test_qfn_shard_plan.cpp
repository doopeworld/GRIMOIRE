// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/qfn_shard_plan.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

using namespace b70;

static_assert(QfnShardPlan::no_local_expert == -1);

static QfnShardShape dense_shape() {
    QfnShardShape s{};
    s.layers = 48;
    for (int l = 0; l < 48; ++l)
        s.layer_kinds[l] = l % 4 == 3 ? QfnLayerKind::attn : QfnLayerKind::gdn;
    s.hidden = 2560;
    s.q_heads = 24;
    s.kv_heads = 2;
    s.attn_head_dim = 256;
    s.q_proj_rows = 12288;
    s.gdn_k_heads = 16;
    s.gdn_v_heads = 48;
    s.gdn_k_head_dim = 128;
    s.gdn_v_head_dim = 128;
    s.gdn_conv_kernel = 4;
    s.routed_experts = 512;
    s.experts_per_token = 10;
    s.routed_width = 640;
    s.shared_width = 640;
    return s;
}

static void expect_span(QfnSpan s, std::int64_t begin, std::int64_t count) {
    assert(s.begin == begin && s.count == count && s.end() == begin + count);
}

// Coordinate-tagged rows of a full-rank tensor, without allocating weights.
static void append_rows(std::vector<int>& rows, QfnSpan span) {
    for (auto i = span.begin; i < span.end(); ++i) rows.push_back(static_cast<int>(i));
}

static void expect_rows(const std::vector<int>& rows, int begin, int count) {
    assert(rows.size() == static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) assert(rows[i] == begin + i);
}

static void reconstruct(QfnSharedLayout layout, const std::array<int, 4>& widths) {
    std::vector<int> queries, gates, output_k, shared_gate, shared_up, shared_k;
    std::array<std::vector<int>, 3> gdn_qkv;
    std::array<std::vector<int>, 2> gdn_ab;
    std::vector<int> gdn_v_channels;
    std::array<int, 2> kv_replicas{};
    std::array<int, 512> expert_owners{};
    int shared_begin = 0;
    for (int r = 0; r < 4; ++r) {
        const char* refusal = "not cleared";
        const auto p = QfnShardPlan::make(dense_shape(), 4, r, layout, refusal);
        assert(p && refusal == nullptr);
        assert(p->rank == r && p->layers == 48 && p->hidden == 2560);
        assert(p->attn_head_dim == 256 && p->gdn_k_head_dim == 128 && p->gdn_v_head_dim == 128);
        assert(p->gdn_conv_kernel == 4);
        expect_span(p->q_heads, r * 6, 6);
        expect_span(p->kv_heads, r / 2, 1);
        expect_span(p->kv_proj_rows, (r / 2) * 256, 256);
        for (auto q = p->q_heads.begin; q < p->q_heads.end(); ++q)
            assert(q / 12 == p->kv_heads.begin);
        ++kv_replicas[p->kv_heads.begin];
        expect_span(p->q_gate_rows, r * 3072, 3072);
        for (auto row = p->q_gate_rows.begin; row < p->q_gate_rows.end(); row += 512) {
            append_rows(queries, {row, 256});
            append_rows(gates, {row + 256, 256});
        }
        expect_span(p->o_proj_k, r * 1536, 1536);
        append_rows(output_k, p->o_proj_k);

        expect_span(p->gdn_k_heads, r * 4, 4);
        expect_span(p->gdn_v_heads, r * 12, 12);
        for (auto v = p->gdn_v_heads.begin; v < p->gdn_v_heads.end(); ++v)
            assert(v / 3 >= p->gdn_k_heads.begin && v / 3 < p->gdn_k_heads.end());
        expect_span(p->gdn_qkv_rows[0], r * 512, 512);
        expect_span(p->gdn_qkv_rows[1], 2048 + r * 512, 512);
        expect_span(p->gdn_qkv_rows[2], 4096 + r * 1536, 1536);
        for (std::size_t i = 0; i < gdn_qkv.size(); ++i) append_rows(gdn_qkv[i], p->gdn_qkv_rows[i]);
        expect_span(p->gdn_ab_rows[0], r * 12, 12);
        expect_span(p->gdn_ab_rows[1], 48 + r * 12, 12);
        for (std::size_t i = 0; i < gdn_ab.size(); ++i) append_rows(gdn_ab[i], p->gdn_ab_rows[i]);
        expect_span(p->gdn_v_channels, r * 1536, 1536);
        append_rows(gdn_v_channels, p->gdn_v_channels);

        expect_span(p->shared_channels, shared_begin, widths[r]);
        expect_span(p->shared_gate_up_rows[0], shared_begin, widths[r]);
        expect_span(p->shared_gate_up_rows[1], 640 + shared_begin, widths[r]);
        if (layout == QfnSharedLayout::g128) {
            assert(p->shared_channels.begin % 128 == 0 && p->shared_channels.count % 128 == 0);
        }
        append_rows(shared_gate, p->shared_gate_up_rows[0]);
        append_rows(shared_up, p->shared_gate_up_rows[1]);
        append_rows(shared_k, p->shared_channels);
        shared_begin += widths[r];

        expect_span(p->routed_experts, r * 128, 128);
        for (int e = 0; e < 512; ++e) {
            const int local = p->local_expert(e);
            if (e / 128 == r) {
                assert(local == e % 128);
                ++expert_owners[e];
            } else {
                assert(local == QfnShardPlan::no_local_expert);
            }
        }
        constexpr std::array<std::int64_t, 5> invalid_ids{
            std::numeric_limits<std::int64_t>::min(), -1, 512, 1LL << 32,
            std::numeric_limits<std::int64_t>::max()};
        for (auto e : invalid_ids) assert(p->local_expert(e) == QfnShardPlan::no_local_expert);
    }
    assert((kv_replicas == std::array<int, 2>{2, 2}));
    for (int n : expert_owners) assert(n == 1);
    assert(queries.size() == 6144 && gates.size() == 6144);
    for (int d = 0; d < 6144; ++d) {
        const int row = (d / 256) * 512 + d % 256;
        assert(queries[d] == row && gates[d] == row + 256);
    }
    expect_rows(output_k, 0, 6144);
    expect_rows(gdn_qkv[0], 0, 2048);
    expect_rows(gdn_qkv[1], 2048, 2048);
    expect_rows(gdn_qkv[2], 4096, 6144);
    expect_rows(gdn_ab[0], 0, 48);
    expect_rows(gdn_ab[1], 48, 48);
    expect_rows(gdn_v_channels, 0, 6144);
    expect_rows(shared_gate, 0, 640);
    expect_rows(shared_up, 640, 640);
    expect_rows(shared_k, 0, 640);
}

static void expect_refused(const QfnShardShape& s, int world, int rank, QfnSharedLayout layout) {
    const char* refusal = nullptr;
    assert(!QfnShardPlan::make(s, world, rank, layout, refusal));
    assert(refusal && *refusal);
}

static void check_refusals() {
    const auto good = dense_shape();
    constexpr auto bf16 = QfnSharedLayout::bf16;
    for (int world : {-1, 0, 1, 2, 3, 5, 8}) expect_refused(good, world, 0, bf16);
    for (int rank : {-1, 4, std::numeric_limits<int>::max()}) expect_refused(good, 4, rank, bf16);
    expect_refused(good, 4, 0, QfnSharedLayout::unsupported);
    expect_refused({}, 4, 0, bf16);

    // Reject shapes even when the changed dimension would still divide evenly.
    for (auto field : {&QfnShardShape::layers, &QfnShardShape::hidden, &QfnShardShape::q_heads,
                       &QfnShardShape::kv_heads, &QfnShardShape::attn_head_dim,
                       &QfnShardShape::q_proj_rows, &QfnShardShape::gdn_k_heads,
                       &QfnShardShape::gdn_v_heads, &QfnShardShape::gdn_k_head_dim,
                       &QfnShardShape::gdn_v_head_dim, &QfnShardShape::gdn_conv_kernel,
                       &QfnShardShape::routed_experts, &QfnShardShape::routed_width,
                       &QfnShardShape::experts_per_token, &QfnShardShape::shared_width}) {
        auto bad = good;
        bad.*field *= 2;
        expect_refused(bad, 4, 0, bf16);
    }
    auto bad = good;
    bad.q_proj_rows /= 2; // Ungated attention is not the target.
    expect_refused(bad, 4, 0, bf16);
    bad = good;
    bad.layer_kinds[3] = QfnLayerKind::unknown;
    expect_refused(bad, 4, 0, bf16);
    bad.layer_kinds[3] = QfnLayerKind::sparse_attn;
    expect_refused(bad, 4, 0, bf16);
    bad.layer_kinds[3] = QfnLayerKind::gdn;
    bad.layer_kinds[2] = QfnLayerKind::attn; // Same counts, wrong layer order.
    expect_refused(bad, 4, 0, bf16);
    bad = good;
    bad.shared_width = 512; // Aligned, but not the 640-wide candidate.
    expect_refused(bad, 4, 0, QfnSharedLayout::g128);
}

int main() {
    reconstruct(QfnSharedLayout::bf16, {160, 160, 160, 160});
    reconstruct(QfnSharedLayout::g128, {256, 128, 128, 128});
    check_refusals();
    std::puts("QfnShardPlan: rank reconstruction and refusals passed");
}
