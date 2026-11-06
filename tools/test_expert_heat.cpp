// test_expert_heat: b70::ExpertHeat (include/b70/expert_heat.hpp) on hand-made routings whose right answer is
// known.  Host only.  Prints PASS / FAIL per case and exits non-zero on any failure.
#include "b70/expert_heat.hpp"

#include <cstdio>
#include <vector>

static int fails = 0;
static void check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "PASS" : "FAIL");
    if (!ok) ++fails;
}

int main() {
    using b70::ExpertHeat;
    using b70::ExpertHeatParams;
    using b70::ExpertSwap;
    const int L = 3, E = 8;
    // resident: experts 0..3 of every layer
    std::vector<std::vector<char>> res(L, std::vector<char>(E, 0));
    for (auto& r : res) for (int e = 0; e < 4; ++e) r[size_t(e)] = 1;
    auto resident = [&](int l, int e) { return res[size_t(l)][size_t(e)] != 0; };
    auto all_movable = [](int, int) { return true; };

    {   // 1. a cold model proposes nothing; one stray route does not pass min_gain
        ExpertHeat h; h.init(L, E, ExpertHeatParams{});
        std::vector<ExpertSwap> s;
        check(h.propose(resident, all_movable, s) == 0 && s.empty(), "no routing -> no swap");
        const int32_t r1[] = {6};
        h.credit(1, r1, 1);
        check(h.propose(resident, all_movable, s) == 0, "one route of a RAM expert < 1.5 -> no swap");
    }
    {   // 2. expert 6 of layer 1 routed every step for 3 steps -> promoted over a zero-heat resident of the
        //    SAME or another layer (the coldest resident, ties by (layer, expert): layer 0 expert 0)
        ExpertHeat h; h.init(L, E, ExpertHeatParams{});
        const int32_t r[] = {6};
        for (int t = 0; t < 3; ++t) h.credit(1, r, 1);
        std::vector<ExpertSwap> s;
        const size_t n = h.propose(resident, all_movable, s);
        check(n == 1 && s[0].promote_layer == 1 && s[0].promote_expert == 6,
              "a repeatedly routed RAM expert is promoted");
        check(n == 1 && s[0].demote_layer == 0 && s[0].demote_expert == 0,
              "the victim is the coldest resident of ANY layer, ties by (layer, expert)");
    }
    {   // 3. hysteresis: a resident at heat 4 is not displaced by a candidate at 5 (needs 4*1.25+1.5 = 6.5)
        ExpertHeatParams p; p.decay = 1.0f;
        ExpertHeat h; h.init(1, E, p);
        std::vector<std::vector<char>> r1(1, std::vector<char>(E, 0));
        r1[0][0] = 1;
        auto res1 = [&](int l, int e) { return r1[size_t(l)][size_t(e)] != 0; };
        const int32_t a[] = {0}, b[] = {5};
        for (int t = 0; t < 4; ++t) h.credit(0, a, 1);
        for (int t = 0; t < 5; ++t) h.credit(0, b, 1);
        std::vector<ExpertSwap> s;
        check(h.propose(res1, all_movable, s) == 0, "hot 5 vs resident 4 -> held (guard 6.5)");
        h.credit(0, b, 1); h.credit(0, b, 1);
        check(h.propose(res1, all_movable, s) == 1 && s[0].promote_expert == 5 && s[0].demote_expert == 0,
              "hot 7 vs resident 4 -> swapped");
    }
    {   // 4. the budget: 10 hot candidates, moves_per_step 3 -> 3 swaps, the 3 hottest
        ExpertHeatParams p; p.decay = 1.0f; p.moves_per_step = 3;
        const int E2 = 32;
        ExpertHeat h; h.init(1, E2, p);
        std::vector<char> r2(E2, 0);
        for (int e = 0; e < 16; ++e) r2[size_t(e)] = 1;
        auto res2 = [&](int, int e) { return r2[size_t(e)] != 0; };
        for (int e = 16; e < 26; ++e) {
            const int32_t id[] = {e};
            for (int t = 0; t < 2 + (e - 16); ++t) h.credit(0, id, 1);
        }
        std::vector<ExpertSwap> s;
        const size_t n = h.propose(res2, all_movable, s);
        check(n == 3 && s[0].promote_expert == 25 && s[1].promote_expert == 24 && s[2].promote_expert == 23,
              "budget 3 -> the three hottest (25, 24, 23)");
    }
    {   // 5. not movable: another rank's experts and copies in flight are never proposed
        ExpertHeat h; h.init(L, E, ExpertHeatParams{});
        const int32_t r[] = {6, 7};
        for (int t = 0; t < 4; ++t) h.credit(2, r, 2);
        auto mov = [](int l, int e) { return !(l == 2 && e == 6) && !(l == 0); };
        std::vector<ExpertSwap> s;
        const size_t n = h.propose(resident, mov, s);
        bool ok = n == 1 && s[0].promote_layer == 2 && s[0].promote_expert == 7 && s[0].demote_layer != 0;
        check(ok, "unmovable candidates and victims are skipped");
    }
    {   // 6. credit cap: a 4096-route prompt call credits 512 in total; decay applies per credit
        ExpertHeatParams p; p.decay = 0.5f;
        ExpertHeat h; h.init(1, E, p);
        std::vector<int32_t> big(4096, 3);
        h.credit(0, big.data(), big.size());
        const bool capped = h.heat(0, 3) > 511.9f && h.heat(0, 3) < 512.1f;
        const int32_t one[] = {-1, 99};      // out of range: skipped
        h.credit(0, one, 2);
        check(capped && h.heat(0, 3) > 255.9f && h.heat(0, 3) < 256.1f,
              "a prompt call credits at most 512; out-of-range ids skipped; decay per credit");
    }
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
