// route_sim: replays a GRIMOIRE_ROUTE_TRACE file against expert placements and reports how many
// routed experts per decode token and layer would come from pinned RAM (the CPU experts that
// set Flash-Next's one-card decode speed).  Host only, no GPU, no engine.
//
//   route_sim TRACE NV [HITS] [token_ms]
//     TRACE     GRIMOIRE_ROUTE_TRACE output: records of int32 {layer, rows, top_k, kind} + rows*top_k
//               int32 expert ids; kind 0 = prompt (streamed), 1 = small-M prompt / verify, 2 = decode
//     NV        experts per layer in VRAM (GRIMOIRE_EXPERT_VRAM_PER_LAYER)
//     HITS      the GRIMOIRE_EXPERT_HITS file the run loaded ([moe layer][expert] uint64); the
//               static placement is the loader's (stable sort by hits, top NV); none = the first NV
//     token_ms  decode time of one token, for the copy-engine model (default 20)
//
// Policies (every one starts from the static placement):
//   static   the loader's placement, never changed (today)
//   oracle   the NV most-routed experts of THIS trace per layer (hindsight; no copies)
//   lru B    after each call of a layer, up to B of its routed RAM experts are copied to VRAM,
//            evicting the least recently routed VRAM experts not routed in that call
//   lfu B g  the same admission, ranked by a score decayed by g per call of the layer; an expert
//            enters only if its score beats the weakest VRAM expert's
// Copies: one at a time on a copy engine (2.76 MB at 12.4 GB/s = 0.223 ms, measured 09-26); an
// admitted expert counts as a VRAM hit only from the first call of its layer that starts after
// its copy ends (the victim leaves VRAM at once -- its RAM copy stays valid, experts never change).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <string>
#include <vector>

struct Rec { int layer, rows, k, kind; std::vector<int32_t> ids; };

static bool read_trace(const char* path, std::vector<Rec>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::perror(path); return false; }
    int32_t h[4];
    while (std::fread(h, sizeof(h), 1, f) == 1) {
        if (h[1] <= 0 || h[1] > 1 << 20 || h[2] <= 0 || h[2] > 64 || h[3] < 0 || h[3] > 2) {
            std::fprintf(stderr, "bad record header {%d %d %d %d} at record %zu\n", h[0], h[1], h[2], h[3],
                         out.size());
            std::fclose(f);
            return false;
        }
        Rec r{h[0], h[1], h[2], h[3], std::vector<int32_t>(static_cast<size_t>(h[1]) * size_t(h[2]))};
        if (std::fread(r.ids.data(), sizeof(int32_t), r.ids.size(), f) != r.ids.size()) {
            std::fprintf(stderr, "truncated record %zu (ignored)\n", out.size());
            break;
        }
        out.push_back(std::move(r));
    }
    std::fclose(f);
    return true;
}

struct Result {
    double ram_per_tl = 0; double swaps_per_tok = 0; long tl = 0; std::vector<long> hist;
    // small-M calls (kind 1: verify rows / short prompts): distinct RAM experts per layer call -- what
    // CpuExperts::ffn_rows reads -- and the rows they served
    double sm_union = 0; double sm_rows = 0; long sm_calls = 0;
};

// One layer's VRAM set: in[e] = resident now; pend = admitted, copy not yet committed.
struct LayerState {
    std::vector<char> in;               // resident
    std::vector<double> last, score;    // last call index routed, decayed score
    std::vector<int> pend_e;            // pending admissions (expert)
    std::vector<double> pend_t;         // their copy end times (ms)
    long calls = 0;
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: route_sim TRACE NV [HITS] [token_ms]\n");
        return 2;
    }
    std::vector<Rec> recs;
    if (!read_trace(argv[1], recs)) return 1;
    const int NV = std::atoi(argv[2]);
    const char* hits_path = argc > 3 && std::strcmp(argv[3], "-") != 0 ? argv[3] : nullptr;
    const double tok_ms = argc > 4 ? std::atof(argv[4]) : 20.0;
    const double copy_ms = 2764864.0 / 12.4e9 * 1e3;

    int E = 0, nl = 0;
    long n_dec = 0, n_ver = 0, n_pr = 0;
    for (const auto& r : recs) {
        for (int32_t e : r.ids) E = std::max(E, int(e) + 1);
        if (r.layer >= 0) nl = std::max(nl, r.layer + 1);
        (r.kind == 2 ? n_dec : r.kind == 1 ? n_ver : n_pr) += r.layer >= 0;
    }
    E = E <= 256 ? 256 : E <= 512 ? 512 : E;
    std::printf("trace: %zu records, %d layers, %d experts, decode %ld / small-M %ld / prompt %ld layer calls\n",
                recs.size(), nl, E, n_dec, n_ver, n_pr);
    if (NV <= 0 || NV > E || nl == 0) { std::fprintf(stderr, "nothing to simulate\n"); return 1; }

    // static placement: the loader's rule (moe layer index = backbone layer index for Flash-Next,
    // whose 48 layers are all MoE)
    std::vector<std::vector<char>> stat(size_t(nl), std::vector<char>(size_t(E), 0));
    {
        std::vector<uint64_t> hits(size_t(nl) * E, 0);
        if (hits_path) {
            FILE* f = std::fopen(hits_path, "rb");
            if (!f) { std::perror(hits_path); return 1; }
            const size_t got = std::fread(hits.data(), sizeof(uint64_t), hits.size(), f);
            std::fclose(f);
            if (got != hits.size()) std::fprintf(stderr, "hits file short: %zu of %zu counts\n", got, hits.size());
        }
        for (int l = 0; l < nl; ++l) {
            std::vector<int> ord(static_cast<size_t>(E));
            for (int e = 0; e < E; ++e) ord[size_t(e)] = e;
            const uint64_t* h = hits.data() + size_t(l) * E;
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return h[a] > h[b]; });
            for (int k = 0; k < NV; ++k) stat[size_t(l)][size_t(ord[size_t(k)])] = 1;
        }
    }
    // oracle: this trace's own counts (every kind)
    std::vector<std::vector<char>> orac(size_t(nl), std::vector<char>(size_t(E), 0));
    {
        std::vector<std::vector<long>> cnt(size_t(nl), std::vector<long>(size_t(E), 0));
        for (const auto& r : recs)
            if (r.layer >= 0)
                for (int32_t e : r.ids) if (e >= 0 && e < E) ++cnt[size_t(r.layer)][size_t(e)];
        for (int l = 0; l < nl; ++l) {
            std::vector<int> ord(static_cast<size_t>(E));
            for (int e = 0; e < E; ++e) ord[size_t(e)] = e;
            const auto& c = cnt[size_t(l)];
            std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return c[size_t(a)] > c[size_t(b)]; });
            for (int k = 0; k < NV; ++k) orac[size_t(l)][size_t(ord[size_t(k)])] = 1;
        }
    }

    // Radiance's heat engine (core/place/heat.h): every layer's heat decays by `decay` each time its
    // routing is credited (a prompt call's credit scaled down to 512 selections); after each step, up to
    // S swaps over the whole model: the hottest non-resident expert of ANY layer replaces the coldest
    // resident one (so a layer may hold more or fewer than NV), only when hot >= cold * 1.25 + 1.5.
    // The swap's copy runs on the copy engine; the expert is resident from the first call after it lands.
    auto run_heat = [&](int S, double decay) {
        Result res;
        res.hist.assign(11, 0);
        std::vector<std::vector<char>> in = stat;
        std::vector<std::vector<double>> heat(size_t(nl), std::vector<double>(size_t(E), 0.0));
        struct Pend { int l, e; double t; };
        std::vector<Pend> pend;
        double now = 0.0, copy_free = 0.0;
        long swaps = 0, ram = 0, tokens = 0;
        int prev_layer = 1 << 30;
        auto heat_step = [&](double t) {
            // candidates: non-resident, not pending; victims: resident
            std::vector<std::pair<double, std::pair<int, int>>> hot, cold;
            for (int l = 0; l < nl; ++l)
                for (int e = 0; e < E; ++e) {
                    if (in[size_t(l)][size_t(e)]) cold.push_back({heat[size_t(l)][size_t(e)], {l, e}});
                    else if (heat[size_t(l)][size_t(e)] > 0) hot.push_back({heat[size_t(l)][size_t(e)], {l, e}});
                }
            for (const auto& p : pend) {
                auto it = std::find_if(hot.begin(), hot.end(), [&](const auto& h) {
                    return h.second.first == p.l && h.second.second == p.e; });
                if (it != hot.end()) hot.erase(it);
            }
            const size_t nh = std::min<size_t>(size_t(S), hot.size()), nc = std::min<size_t>(size_t(S), cold.size());
            std::partial_sort(hot.begin(), hot.begin() + long(nh), hot.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            std::partial_sort(cold.begin(), cold.begin() + long(nc), cold.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
            for (size_t i = 0; i < nh && i < nc; ++i) {
                if (!(hot[i].first >= cold[i].first * 1.25 + 1.5)) break;
                in[size_t(cold[i].second.first)][size_t(cold[i].second.second)] = 0;
                const double start = std::max(copy_free, t);
                copy_free = start + copy_ms;
                pend.push_back({hot[i].second.first, hot[i].second.second, copy_free});
                ++swaps;
            }
        };
        for (const auto& r : recs) {
            if (r.layer < 0) continue;
            if (r.layer <= prev_layer) {
                if (prev_layer != (1 << 30)) heat_step(now + tok_ms);
                now += tok_ms;
                if (r.kind == 2) ++tokens;
            }
            prev_layer = r.layer;
            const double t_call = now + tok_ms * r.layer / nl;
            for (size_t j = 0; j < pend.size();) {
                if (pend[j].t <= t_call) { in[size_t(pend[j].l)][size_t(pend[j].e)] = 1; pend.erase(pend.begin() + long(j)); }
                else ++j;
            }
            auto& s_in = in[size_t(r.layer)];
            std::vector<int> miss;
            for (int row = 0; row < r.rows; ++row) {
                int n = 0;
                for (int k = 0; k < r.k; ++k) {
                    const int e = r.ids[size_t(row) * r.k + k];
                    if (e < 0 || e >= E || s_in[size_t(e)]) continue;
                    ++n;
                    if (std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
                }
                if (r.kind == 2) { ram += n; ++res.tl; ++res.hist[size_t(std::min(n, 10))]; }
            }
            if (r.kind == 1) { res.sm_union += double(miss.size()); res.sm_rows += r.rows; ++res.sm_calls; }
            // credit: decay the layer's plane, add this call's selections (capped at 512 in total)
            auto& h = heat[size_t(r.layer)];
            for (double& x : h) x *= decay;
            const double total = double(r.ids.size());
            const double w = total > 512.0 ? 512.0 / total : 1.0;
            for (int32_t e : r.ids) if (e >= 0 && e < E) h[size_t(e)] += w;
        }
        res.ram_per_tl = res.tl ? double(ram) / double(res.tl) : 0.0;
        res.swaps_per_tok = tokens ? double(swaps) / double(tokens) : 0.0;
        return res;
    };

    // policy: 0 static, 1 oracle, 2 lru, 3 lfu
    auto run = [&](int pol, int B, double g) {
        Result res;
        res.hist.assign(11, 0);
        std::vector<LayerState> st(static_cast<size_t>(nl));
        for (int l = 0; l < nl; ++l) {
            st[size_t(l)].in = pol == 1 ? orac[size_t(l)] : stat[size_t(l)];
            st[size_t(l)].last.assign(size_t(E), -1e18);
            st[size_t(l)].score.assign(size_t(E), 0.0);
        }
        double now = 0.0, copy_free = 0.0;    // ms; the copy engine is free from copy_free on
        long swaps = 0, ram = 0, tokens = 0;
        int prev_layer = 1 << 30;
        for (const auto& r : recs) {
            if (r.layer < 0) continue;
            LayerState& s = st[size_t(r.layer)];
            // time: a decode token's layers are spread over tok_ms; a new token starts when the
            // layer index wraps (prompt / verify calls advance the clock the same way)
            if (r.layer <= prev_layer) { now += tok_ms; if (r.kind == 2) ++tokens; }
            prev_layer = r.layer;
            const double t_call = now + tok_ms * r.layer / nl;
            // commit copies that ended before this call
            for (size_t j = 0; j < s.pend_e.size();) {
                if (s.pend_t[j] <= t_call) {
                    s.in[size_t(s.pend_e[j])] = 1;
                    s.pend_e.erase(s.pend_e.begin() + long(j));
                    s.pend_t.erase(s.pend_t.begin() + long(j));
                } else ++j;
            }
            // count: RAM experts per row of this call (decode rows only enter the result)
            std::vector<int> miss;
            for (int row = 0; row < r.rows; ++row) {
                int n = 0;
                for (int k = 0; k < r.k; ++k) {
                    const int e = r.ids[size_t(row) * r.k + k];
                    if (e < 0 || e >= E) continue;
                    if (!s.in[size_t(e)]) {
                        ++n;
                        if (std::find(miss.begin(), miss.end(), e) == miss.end()) miss.push_back(e);
                    }
                }
                if (r.kind == 2) { ram += n; ++res.tl; ++res.hist[size_t(std::min(n, 10))]; }
            }
            if (r.kind == 1) { res.sm_union += double(miss.size()); res.sm_rows += r.rows; ++res.sm_calls; }
            // history
            const double ci = double(s.calls++);
            for (int32_t e : r.ids) if (e >= 0 && e < E) { s.last[size_t(e)] = ci; }
            if (pol == 3) {
                for (int e = 0; e < E; ++e) s.score[size_t(e)] *= g;
                for (int32_t e : r.ids) if (e >= 0 && e < E) s.score[size_t(e)] += 1.0;
            }
            if (pol < 2 || miss.empty()) continue;
            // admission: up to B of this call's RAM experts not already pending
            std::vector<char> routed(size_t(E), 0);
            for (int32_t e : r.ids) if (e >= 0 && e < E) routed[size_t(e)] = 1;
            if (pol == 3)
                std::sort(miss.begin(), miss.end(), [&](int a, int b) { return s.score[size_t(a)] > s.score[size_t(b)]; });
            int admitted = 0;
            for (int e : miss) {
                if (admitted >= B) break;
                if (std::find(s.pend_e.begin(), s.pend_e.end(), e) != s.pend_e.end()) continue;
                // victim: resident, not routed now, least recent (lru) / lowest score (lfu)
                int v = -1;
                for (int x = 0; x < E; ++x) {
                    if (!s.in[size_t(x)] || routed[size_t(x)]) continue;
                    if (v < 0) { v = x; continue; }
                    const bool better = pol == 2 ? s.last[size_t(x)] < s.last[size_t(v)]
                                                 : s.score[size_t(x)] < s.score[size_t(v)];
                    if (better) v = x;
                }
                if (v < 0) break;
                if (pol == 3 && !(s.score[size_t(e)] > s.score[size_t(v)])) continue;
                s.in[size_t(v)] = 0;
                const double start = std::max(copy_free, t_call);
                copy_free = start + copy_ms;
                s.pend_e.push_back(e);
                s.pend_t.push_back(copy_free);
                ++swaps; ++admitted;
            }
        }
        res.ram_per_tl = res.tl ? double(ram) / double(res.tl) : 0.0;
        res.swaps_per_tok = tokens ? double(swaps) / double(tokens) : 0.0;
        return res;
    };

    auto show = [&](const char* name, const Result& r) {
        std::printf("%-22s RAM experts / decode token-layer %.3f  swaps / token %.1f  (%ld token-layers)  hist",
                    name, r.ram_per_tl, r.swaps_per_tok, r.tl);
        for (size_t i = 0; i < r.hist.size(); ++i) std::printf(" %ld", r.hist[i]);
        if (r.sm_calls)
            std::printf("  | small-M: %.2f distinct RAM experts / layer call (%.1f rows, %ld calls)",
                        r.sm_union / double(r.sm_calls), r.sm_rows / double(r.sm_calls), r.sm_calls);
        std::printf("\n");
    };
    std::printf("NV %d, token %.1f ms, copy %.3f ms an expert\n", NV, tok_ms, copy_ms);
    show("static (loader)", run(0, 0, 0));
    show("oracle (hindsight)", run(1, 0, 0));
    for (int B : {1, 2, 4}) {
        char n[64];
        std::snprintf(n, sizeof n, "lru B=%d", B);
        show(n, run(2, B, 0));
    }
    for (int B : {1, 2}) for (double g : {0.9, 0.98, 0.995}) {
        char n[64];
        std::snprintf(n, sizeof n, "lfu B=%d g=%.3f", B, g);
        show(n, run(3, B, g));
    }
    for (int S : {16, 32, 64}) {
        char n[64];
        std::snprintf(n, sizeof n, "heat S=%d (Radiance)", S);
        show(n, run_heat(S, 0.95));
    }
    return 0;
}
