// expert_heat.hpp -- which tiered experts should be in VRAM, decided from the router's choices.
//
// The policy of Radiance's heat engine (codeberg StillDeadcode/radiance, core/place/heat.h,
// Apache-2.0), rewritten for GRIMOIRE's tiered experts: a heat per (layer, expert) that decays
// each time its layer's routing is credited, and after each step a few swaps over the WHOLE model
// -- the hottest expert not in VRAM, of any layer, replaces the coldest one that is -- guarded by
// a ratio and an additive margin so the two ends of the ranking do not trade places on noise.
// Radiance's replay (Qwen3.8-Flash-Next TP1, 40% of experts resident): 78.0% of routes resident
// at 64 swaps a step, an oracle at the same budget 78.4%, static profile placement far lower.
//
// Host code only, no device calls: the engine (grimoire.cpp, the mover) owns the copies and the
// tables; this owns the decision, so tools/test_expert_heat.cpp and tools/route_sim.cpp can drive
// exactly what the engine runs.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace b70 {

struct ExpertHeatParams {
    // applied to a layer's heats each time that layer's routing is credited; 0.95 = a half-life of
    // ~14 steps (Radiance's replay: flat from 0.8 to 0.95, worse above -- a long memory keeps
    // experts the conversation has moved past)
    float decay = 0.95f;
    // promote only when hot >= cold * hysteresis + min_gain (min_gain stops swaps against a
    // zero-heat victim, where any ratio passes)
    float hysteresis = 1.25f;
    float min_gain = 1.5f;
    // the most one call credits, in selections: a prompt chunk routes thousands of rows and would
    // otherwise make the resident set the prompt's for dozens of steps (Radiance: decode hit rate
    // 77.3% uncapped, 79.0% at 512)
    float credit_cap = 512.0f;
    // swaps proposed per step (each = one expert block over PCIe each way, 2.76 MB at ~12 GB/s on
    // a Gen4 x8 B70 = ~0.23 ms)
    int moves_per_step = 32;
};

struct ExpertSwap {
    int promote_layer = -1, promote_expert = -1;   // to VRAM
    int demote_layer = -1, demote_expert = -1;     // to pinned RAM
};

class ExpertHeat {
public:
    void init(int layers, int experts, const ExpertHeatParams& p) {
        L_ = layers; E_ = experts; P_ = p;
        h_.assign(static_cast<size_t>(layers) * static_cast<size_t>(experts), 0.0f);
    }
    bool ready() const { return L_ > 0 && E_ > 0; }
    const ExpertHeatParams& params() const { return P_; }
    float heat(int l, int e) const { return h_[idx(l, e)]; }

    // One call of layer l: its n routes (expert ids; out-of-range ids, e.g. -1, are skipped).
    void credit(int l, const int32_t* ids, size_t n) {
        if (l < 0 || l >= L_ || !ids) return;
        float* row = h_.data() + idx(l, 0);
        for (int e = 0; e < E_; ++e) row[e] *= P_.decay;
        const float w = n > P_.credit_cap ? P_.credit_cap / float(n) : 1.0f;
        for (size_t i = 0; i < n; ++i)
            if (ids[i] >= 0 && ids[i] < E_) row[ids[i]] += w;
    }

    // Swaps for this step.  resident(l, e) -> bool: in VRAM now; movable(l, e) -> bool: may change tier
    // (false for another rank's experts and for those with a copy in flight).  Appends at most
    // moves_per_step swaps, the hottest candidates against the coldest victims, while the guard holds.
    template <class Resident, class Movable>
    size_t propose(Resident resident, Movable movable, std::vector<ExpertSwap>& out) const {
        struct U { float h; int l, e; };
        std::vector<U> hot, cold;
        for (int l = 0; l < L_; ++l)
            for (int e = 0; e < E_; ++e) {
                if (!movable(l, e)) continue;
                const float x = h_[idx(l, e)];
                if (resident(l, e)) cold.push_back({x, l, e});
                else if (x > 0.0f) hot.push_back({x, l, e});
            }
        const size_t S = static_cast<size_t>(std::max(0, P_.moves_per_step));
        const size_t nh = std::min(S, hot.size()), nc = std::min(S, cold.size());
        // ties broken by (layer, expert) so the choice never depends on the sort's internals
        auto hotter = [](const U& a, const U& b) {
            return a.h != b.h ? a.h > b.h : (a.l != b.l ? a.l < b.l : a.e < b.e); };
        auto colder = [](const U& a, const U& b) {
            return a.h != b.h ? a.h < b.h : (a.l != b.l ? a.l < b.l : a.e < b.e); };
        std::partial_sort(hot.begin(), hot.begin() + static_cast<std::ptrdiff_t>(nh), hot.end(), hotter);
        std::partial_sort(cold.begin(), cold.begin() + static_cast<std::ptrdiff_t>(nc), cold.end(), colder);
        size_t n = 0;
        for (size_t i = 0; i < nh && i < nc; ++i) {
            if (!(hot[i].h >= cold[i].h * P_.hysteresis + P_.min_gain)) break;
            out.push_back({hot[i].l, hot[i].e, cold[i].l, cold[i].e});
            ++n;
        }
        return n;
    }

private:
    size_t idx(int l, int e) const { return static_cast<size_t>(l) * static_cast<size_t>(E_) + static_cast<size_t>(e); }
    int L_ = 0, E_ = 0;
    ExpertHeatParams P_;
    std::vector<float> h_;
};

}  // namespace b70
