// SPDX-License-Identifier: GPL-3.0-or-later
// Included by tools/inkling2_gpu.cpp inside namespace ink, after class Engine.
// Small-M kernels (verify blocks of up to 8 rows), verify commit, and the DSpark drafter
// (mtp.safetensors; reference: SGLang models/inkling2_mtp.py, dflash.py, dspark.py).

// Debug dumps (INK_DUMP=dir): raw device bytes to dir/name.bin
static void dump_dev(sycl::queue& q, const char* name, const void* p, size_t bytes) {
    static const char* dir = std::getenv("INK_DUMP");
    if (!dir) return;
    std::vector<char> h(bytes);
    q.memcpy(h.data(), p, bytes).wait();
    std::ofstream f(std::string(dir) + "/" + name + ".bin", std::ios::binary);
    f.write(h.data(), std::streamsize(bytes));
}

// y[m][r] for m < M <= 8: every weight row read once for all rows.
template <int MODE>
void Engine::gemvm(const uint16_t* W, int rows, int K, const float* x, size_t ldx, float* y, size_t ldy, int M, float cap) {
    if (M < 1 || M > 8) throw std::runtime_error("gemvm: 1..8 rows");
    const size_t groups = (size_t(rows) + 7) / 8;
    q_.parallel_for<InkGemvM<MODE>>(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        const int r = int(it.get_group(0)) * 8 + int(sg.get_group_linear_id());
        if (r >= rows) return;
        const uint16_t* w = W + size_t(r) * K;
        const int l = int(sg.get_local_linear_id());
        float acc[8];
#pragma unroll
        for (int m = 0; m < 8; ++m) acc[m] = 0.f;
        for (int k = l * 8; k < K; k += 128) {
            const sycl::vec<uint16_t, 8> wv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(w + k);
            float wf[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) wf[e] = bfw(wv[e]);
#pragma unroll
            for (int m = 0; m < 8; ++m)
                if (m < M) {
                    const float* xr = x + m * ldx + k;
                    const sycl::float4 a = *reinterpret_cast<const sycl::float4*>(xr);
                    const sycl::float4 b = *reinterpret_cast<const sycl::float4*>(xr + 4);
                    float s = acc[m];
                    s = sycl::fma(wf[0], a[0], s); s = sycl::fma(wf[1], a[1], s); s = sycl::fma(wf[2], a[2], s); s = sycl::fma(wf[3], a[3], s);
                    s = sycl::fma(wf[4], b[0], s); s = sycl::fma(wf[5], b[1], s); s = sycl::fma(wf[6], b[2], s); s = sycl::fma(wf[7], b[3], s);
                    acc[m] = s;
                }
        }
#pragma unroll
        for (int m = 0; m < 8; ++m)
            if (m < M) {
                const float s = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                if (l == 0) {
                    float* o = y + m * ldy + r;
                    if constexpr (MODE == 0) *o = bfr(s);
                    else if constexpr (MODE == 1) *o = s;
                    else { const float lin = bfr(s); *o = cap * sycl::tanh(lin / cap); }
                }
            }
    });
}

// Experts for M <= 8 rows: (row, slot) pairs grouped by expert on the device, each used expert's
// weights read once for all its rows (same per-weight arithmetic as experts_decode).
void Engine::experts_small(const Layer& Ly, int M) {
    const int h = h_, I = c_.inter, TK = c_.top_k, E = c_.experts, e0 = e0_, en = en_;
    const bool tp = world_ > 1;
    const float gcap = c_.gate_cap, ucap = c_.up_cap;
    const int* sel = sel_;
    int* gi = g_info_;  // [0] nd, [1..128] eid, [129..257] start, [258..385] prow, [386..513] ppos
    static const bool old_group = std::getenv("INK_OLD_GROUP") != nullptr;
    if (old_group || E > 128 || M * TK > 128) {
    q_.single_task<InkGroup>([=]() {
        int cnt[128], slot[128], cur[128];
        for (int e = 0; e < E; ++e) cnt[e] = 0;
        for (int t = 0; t < M * TK; ++t) cnt[sel[t]]++;
        int nd = 0, acc = 0;
        for (int e = 0; e < E; ++e)
            if (cnt[e]) { gi[1 + nd] = e; gi[129 + nd] = acc; cur[nd] = acc; slot[e] = nd; acc += cnt[e]; ++nd; }
        gi[129 + nd] = acc;
        gi[0] = nd;
        for (int m = 0; m < M; ++m)
            for (int s = 0; s < TK; ++s) {
                const int u = slot[sel[m * TK + s]];
                const int q = cur[u]++;
                gi[258 + q] = m;
                gi[386 + m * TK + s] = q;
                gi[514 + q] = u;
            }
    });
    } else {
    // the same tables from one 128-thread work-group: counts, used-expert list and slot starts by group scans, each
    // (row, choice) its slot = start + earlier (row, choice) pairs on the same expert (the single_task's order)
    const int NT = M * TK;
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int, 1> cnt(sycl::range<1>(128), cgh), start(sycl::range<1>(128), cgh), ord(sycl::range<1>(128), cgh), sl(sycl::range<1>(128), cgh);
        cgh.parallel_for<InkGroupP>(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) {
            auto g = it.get_group();
            const int e = int(it.get_local_id(0));
            cnt[e] = 0;
            if (e < NT) sl[e] = sel[e];
            sycl::group_barrier(g);
            int c = 0;
            if (e < E) for (int t = 0; t < NT; ++t) c += sl[t] == e;
            const int used = c > 0 ? 1 : 0;
            const int u = sycl::exclusive_scan_over_group(g, used, sycl::plus<int>());
            const int st = sycl::exclusive_scan_over_group(g, c, sycl::plus<int>());
            const int nd = sycl::reduce_over_group(g, used, sycl::plus<int>());
            cnt[e] = c; start[e] = st; ord[e] = u;
            if (used) { gi[1 + u] = e; gi[129 + u] = st; }
            if (e == 0) { gi[0] = nd; gi[129 + nd] = NT; }
            sycl::group_barrier(g);
            if (e < NT) {
                const int ex = sl[e];
                int before = 0;
                for (int t = 0; t < e; ++t) before += sl[t] == ex;
                const int q = start[ex] + before;
                gi[258 + q] = e / TK;
                gi[386 + e] = q;
                gi[514 + q] = ord[ex];
            }
        });
    });
    }
    float* act = actS_;
    float* dbuf = dbufS_;
    const float* xbase = row(NM);
    const int U = std::min(E, M * TK);
    constexpr int R = 2, SGS = 8;
    static const bool smes = !std::getenv("INK_NO_SMES");   // GRIMOIRE moe_mxfp4_esimd_small design, NVFP4 / MXFP8 port
    const bool es_ok = smes && h % 128 == 0 && I % 128 == 0 && (2 * I) % 256 == 0 && h % 256 == 0;
    if (es_ok) {
        uint16_t* xg = xgS_;
        const int slots = M * TK;
        static const bool sm_perm = !std::getenv("INK_SM_NOPERM");
        const bool xperm = sm_perm && Ly.fgu == PackedFormat::NVFP4;
        q_.parallel_for<InkGatherS>(sycl::range<1>(size_t(slots) * (h / 8)), [=](sycl::id<1> id) {
            const size_t t = id[0], qd = t / (h / 8), c = (t % (h / 8)) * 8;
            const float* src = xbase + size_t(gi[258 + qd]) * h + c;
            sycl::vec<sycl::half, 8> o;   // bf16 values, exact in fp16; NVFP4 layers: K order 0 4 1 5 2 6 3 7 in each 8
#pragma unroll
            for (int e = 0; e < 8; ++e) o[e] = sycl::half(src[xperm ? ((e & 1) ? 4 + (e >> 1) : (e >> 1)) : e]);
            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(xg + qd * h + c) = o;
        });
        const int fg = Ly.fgu == PackedFormat::NVFP4 ? 0 : 1, fd = Ly.fdn == PackedFormat::NVFP4 ? 0 : 1;
        static const int sm_alu0 = (std::getenv("INK_SM_LUT") ? 0 : 1) | (std::getenv("INK_SM_WHOLE") ? 2 : 0) | (std::getenv("INK_SM_NODBUF") ? 0 : 18);
        const int sm_alu = sm_alu0 | (xperm ? 4 : 0) | (sm_perm && Ly.fdn == PackedFormat::NVFP4 ? 8 : 0);
        const int sm_alu_dn = sm_alu0 | (sm_perm && Ly.fdn == PackedFormat::NVFP4 ? 4 : 0);
        static const bool sm_stat = std::getenv("INK_SM_STAT") != nullptr;   // debug: distinct experts per verify visit
        if (sm_stat) {
            static long calls = 0, sum = 0, rows = 0;
            int nd = 0;
            q_.memcpy(&nd, gi, sizeof(int)).wait();
            ++calls; sum += nd; rows += M;
            if (calls % 480 == 0) std::printf("smstat: %ld verify visits, %.1f distinct experts per visit, %.2f rows\n", calls, double(sum) / calls, double(rows) / calls);
        }
        ink_moe_small_esimd(q_, InkSmArgs{Ly.gu, Ly.gus, 2 * I, h, wen_ * 2 * I, xg, slots, actSb_, gi, e0, en, Ly.gua, Ly.gud, nvlut16_, gcap, ucap, U, sm_alu, own_}, fg, 1);
        ink_moe_small_esimd(q_, InkSmArgs{Ly.dn, Ly.dns, h, I, wen_ * h, actSb_, slots, dbuf, gi, e0, en, Ly.dna, Ly.dnd, nvlut16_, 0.f, 0.f, U, sm_alu_dn, own_}, fd, 2);
    }
    if (!es_ok && rep_) throw std::runtime_error("verify experts: replicated experts need the ESIMD kernels");
    if (!es_ok) {
    {
        const bool NV = Ly.fgu == PackedFormat::NVFP4;
        const uint8_t *W = Ly.gu, *S = Ly.gus;
        const float *amax = Ly.gua, *den = Ly.gud;
        const size_t rb = NV ? h / 2 : h, sb = NV ? h / 16 : h / 32;
        const int steps = h / 256, wpb = I / (R * SGS);
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(32, cgh), wts(16, cgh);
            cgh.parallel_for<InkGuS>(sycl::nd_range<1>(size_t(U) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                const int u = int(it.get_group(0)) / wpb;
                if (u >= gi[0]) return;
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int i0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = gi[1 + u], p0 = gi[129 + u], np = gi[130 + u] - p0;
                if (e < e0 || e >= e0 + en) return;
                const int el = e - e0;
                const float outer = NV ? amax[e] / den[0] : 1.f;
                float g[8][R], uu[8][R];
#pragma unroll
                for (int q = 0; q < 8; ++q)
#pragma unroll
                    for (int r = 0; r < R; ++r) { g[q][r] = 0.f; uu[q][r] = 0.f; }
                for (int k = 0; k < steps; ++k) {
                    const int blk = k * 16 + lane;
                    sycl::uint4 pg[R], pu[R];
                    float qg1[R], qg3[R], qu1[R], qu3[R];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t gr = size_t(el) * 2 * I + 2 * size_t(i0 + r), ur = gr + 1;
                        if (NV) {
                            const sycl::uint2 a = *reinterpret_cast<const sycl::uint2*>(W + gr * rb + size_t(blk) * 8);
                            const sycl::uint2 b = *reinterpret_cast<const sycl::uint2*>(W + ur * rb + size_t(blk) * 8);
                            pg[r] = sycl::uint4(a[0], a[1], 0u, 0u); pu[r] = sycl::uint4(b[0], b[1], 0u, 0u);
                            const float sgc = e4[S[gr * sb + blk]], suc = e4[S[ur * sb + blk]];
                            qg1[r] = bfrf(sgc * outer); qg3[r] = bfrf((1.5f * sgc) * outer);
                            qu1[r] = bfrf(suc * outer); qu3[r] = bfrf((1.5f * suc) * outer);
                        } else {
                            pg[r] = *reinterpret_cast<const sycl::uint4*>(W + gr * rb + size_t(blk) * 16);
                            pu[r] = *reinterpret_cast<const sycl::uint4*>(W + ur * rb + size_t(blk) * 16);
                            qg1[r] = b70::e8m0_to_f32(S[gr * sb + blk / 2]); qu1[r] = b70::e8m0_to_f32(S[ur * sb + blk / 2]);
                            qg3[r] = qu3[r] = 0.f;
                        }
                    }
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        if (q >= np) break;
                        float xr[16];
                        load_x(xbase + size_t(gi[258 + p0 + q]) * h + size_t(blk) * 16, xr, 16);
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (NV) {
                                float ag = 0.f, bg = 0.f, au = 0.f, bu = 0.f;
                                nv8(pg[r][0], xr, wp, wt, ag, bg); nv8(pg[r][1], xr + 8, wp, wt, ag, bg);
                                nv8(pu[r][0], xr, wp, wt, au, bu); nv8(pu[r][1], xr + 8, wp, wt, au, bu);
                                g[q][r] += qg1[r] * ag + qg3[r] * bg;
                                uu[q][r] += qu1[r] * au + qu3[r] * bu;
                            } else {
                                g[q][r] += (mx4(pg[r][0], xr, e4) + mx4(pg[r][1], xr + 4, e4) + mx4(pg[r][2], xr + 8, e4) + mx4(pg[r][3], xr + 12, e4)) * qg1[r];
                                uu[q][r] += (mx4(pu[r][0], xr, e4) + mx4(pu[r][1], xr + 4, e4) + mx4(pu[r][2], xr + 8, e4) + mx4(pu[r][3], xr + 12, e4)) * qu1[r];
                            }
                        }
                    }
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    if (q >= np) break;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        float gg = sycl::reduce_over_group(sg, g[q][r], sycl::plus<float>());
                        float vv = sycl::reduce_over_group(sg, uu[q][r], sycl::plus<float>());
                        if (lane == 0) {
                            gg = bfrf(gg); vv = bfrf(vv);
                            act[size_t(p0 + q) * I + i0 + r] = bfrf((gcap * sycl::tanh(gg / gcap)) * sigm(gg) * (ucap * sycl::tanh(vv / ucap)));
                        }
                    }
                }
            });
        });
    }
    {
        const bool NV = Ly.fdn == PackedFormat::NVFP4;
        const uint8_t *W = Ly.dn, *S = Ly.dns;
        const float *amax = Ly.dna, *den = Ly.dnd;
        const size_t rb = NV ? I / 2 : I, sb = NV ? I / 16 : I / 32;
        const int steps = I / 128, wpb = h / (R * SGS);
        q_.submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> e4s(256, cgh), wps(32, cgh), wts(16, cgh);
            cgh.parallel_for<InkDownS>(sycl::nd_range<1>(size_t(U) * wpb * SGS * 16, SGS * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int lid = int(it.get_local_id(0));
                float* e4 = e4s.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wp = wps.template get_multi_ptr<sycl::access::decorated::no>().get();
                float* wt = wts.template get_multi_ptr<sycl::access::decorated::no>().get();
                fill_tables(e4, wp, wt, lid, SGS * 16);
                sycl::group_barrier(it.get_group());
                const int u = int(it.get_group(0)) / wpb;
                if (u >= gi[0]) return;
                auto sg = it.get_sub_group();
                const int lane = int(sg.get_local_linear_id());
                const int j0 = (int(it.get_group(0)) % wpb) * R * SGS + int(sg.get_group_linear_id()) * R;
                const int e = gi[1 + u], p0 = gi[129 + u], np = gi[130 + u] - p0;
                if (e < e0 || e >= e0 + en) return;
                const int el = e - e0;
                const float outer = NV ? amax[e] / den[0] : 1.f;
                float d[8][R];
#pragma unroll
                for (int q = 0; q < 8; ++q)
#pragma unroll
                    for (int r = 0; r < R; ++r) d[q][r] = 0.f;
                for (int k = 0; k < steps; ++k) {
                    const int c = k * 16 + lane;
                    sycl::uint2 pw[R];
                    float q1[R], q3[R];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const size_t rw = size_t(el) * h + j0 + r;
                        if (NV) {
                            pw[r] = sycl::uint2(*reinterpret_cast<const uint32_t*>(W + rw * rb + size_t(c) * 4), 0u);
                            const float sc = e4[S[rw * sb + c / 2]];
                            q1[r] = bfrf(sc * outer); q3[r] = bfrf((1.5f * sc) * outer);
                        } else {
                            pw[r] = *reinterpret_cast<const sycl::uint2*>(W + rw * rb + size_t(c) * 8);
                            q1[r] = b70::e8m0_to_f32(S[rw * sb + c / 4]); q3[r] = 0.f;
                        }
                    }
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        if (q >= np) break;
                        float xr[8];
                        load_x(act + size_t(p0 + q) * I + size_t(c) * 8, xr, 8);
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            if (NV) { float A = 0.f, B = 0.f; nv8(pw[r][0], xr, wp, wt, A, B); d[q][r] += q1[r] * A + q3[r] * B; }
                            else d[q][r] += (mx4(pw[r][0], xr, e4) + mx4(pw[r][1], xr + 4, e4)) * q1[r];
                        }
                    }
                }
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    if (q >= np) break;
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const float dd = sycl::reduce_over_group(sg, d[q][r], sycl::plus<float>());
                        if (lane == 0) dbuf[size_t(p0 + q) * h + j0 + r] = bfrf(dd);
                    }
                }
            });
        });
    }
    }
    {   // YB[m] = bf(sum_s w_s d_s) in slot (= ascending expert) order
        const float* w = wts_;
        float* yb = row(YB);
        const int own = own_;
        q_.parallel_for<InkCombS>(sycl::range<1>(size_t(M) * h), [=](sycl::id<1> id) {
            const size_t t = id[0], m = t / h, ch = t % h;
            float acc = 0.f;
            for (int s = 0; s < TK; ++s) {
                const int e = sel[m * TK + s];
                const int q = gi[386 + m * TK + s];
                if (own >= 0 ? (gi[514 + q] & 1) == own : (e >= e0 && e < e0 + en)) acc += w[m * TK + s] * dbuf[size_t(q) * h + ch];
            }
            yb[t] = tp ? acc : bfr(acc);
        });
    }
}

// After a verify block of M rows at positions p..: keep rows 0..a.  The short-conv rings get the
// inputs of positions p + a - 2 .. p + a back from the verify buffer (older entries were not touched).
void Engine::commit_verify(int p, int a) {
    const SlotPtrs* sp = slotptr_;
    const float* vb = vbuf_;
    const int h = h_, S = int(S_.size());
    q_.parallel_for<InkCommit>(sycl::range<1>(size_t(S) * 4 * 3 * h), [=](sycl::id<1> id) {
        const size_t t = id[0];
        const int ch = int(t % h), j = int((t / h) % 3), kind = int((t / (size_t(h) * 3)) % 4), s = int(t / (size_t(h) * 12));
        const int n = sp[s].n[kind];
        if (ch >= n) return;
        const int qpos = p + a + 1 - 3 + j;
        if (qpos < p) return;
        sp[s].ring[kind][size_t(qpos % 3) * n + ch] = vb[((size_t(s) * 4 + kind) * 8 + (qpos - p)) * h + ch];
    });
}

// ---------------------------------------------------------------- DSpark drafter
void Engine::load_mtp(const std::string& path) {
    auto st = std::make_shared<b70::SafeTensors>();
    std::string err;
    if (!st->open(path, err)) throw std::runtime_error("mtp: " + err);
    auto H = [&](const std::string& n, size_t want) {
        const auto* t = st->find(n);
        if (!t) throw std::runtime_error("mtp: missing " + n);
        if (want && size_t(t->end - t->begin) != want) throw std::runtime_error("mtp: size " + n);
        return std::make_pair(static_cast<const uint8_t*>(st->data(*t)), size_t(t->end - t->begin));
    };
    auto up16 = [&](const std::string& n, size_t elems) {
        auto [p, nb] = H(n, elems * 2);
        uint16_t* d = dev_.alloc<uint16_t>(elems);
        q_.memcpy(d, p, nb);
        return d;
    };
    auto cat16 = [&](std::vector<std::pair<std::string, size_t>> parts) {
        size_t total = 0;
        for (auto& pr : parts) total += pr.second;
        uint16_t* d = dev_.alloc<uint16_t>(total);
        size_t at = 0;
        for (auto& pr : parts) { auto [p, nb] = H(pr.first, pr.second * 2); q_.memcpy(d + at, p, nb); at += pr.second; }
        return d;
    };
    const int h = h_, V = c_.output_vocab, Vp = c_.vocab;
    // the drafter needs every vocabulary row: one card has them all; two cards load a full copy for it
    const uint16_t* Urow = unembed_;
    uint16_t* Ufull = nullptr;
    if (world_ > 1) {
        const uint16_t* src = reinterpret_cast<const uint16_t*>(host("model.llm.unembed.weight"));
        Ufull = sycl::malloc_device<uint16_t>(size_t(Vp) * h, q_);
        if (!Ufull) throw std::runtime_error("mtp: full unembed");
        q_.memcpy(Ufull, src, size_t(Vp) * h * 2).wait();
        unembedFullT_ = tiled(src, Vp, h);
        Urow = Ufull;
    } else unembedFullT_ = unembedT_;
    for (int l = 0; l < 3; ++l) {
        const auto s = "layers." + std::to_string(l) + ".";
        DLayer& L = d_.L[l];
        L.sliding = l != 1;
        auto hcat = [&](std::vector<std::pair<std::string, size_t>> parts) {
            std::vector<uint16_t> v;
            for (auto& pr : parts) { auto [pp, nb] = H(pr.first, pr.second * 2); const auto* u = reinterpret_cast<const uint16_t*>(pp); v.insert(v.end(), u, u + pr.second); }
            return v;
        };
        {
            const auto v = hcat({{s + "self_attn.q_proj.weight", size_t(3072) * h}, {s + "self_attn.k_proj.weight", size_t(1024) * h}, {s + "self_attn.v_proj.weight", size_t(1024) * h}});
            L.Wqkv = tiled(v.data(), 5120, h);
        }
        L.Wkvc = cat16({{s + "self_attn.k_proj_ctx.weight", size_t(1024) * h}, {s + "self_attn.v_proj_ctx.weight", size_t(1024) * h}});
        { const auto v = hcat({{s + "self_attn.o_proj.weight", size_t(h) * 3072}}); L.Wo = tiled(v.data(), h, 3072); }
        { const auto v = hcat({{s + "mlp.gate_proj.weight", size_t(9216) * h}, {s + "mlp.up_proj.weight", size_t(9216) * h}}); L.Wgu = tiled(v.data(), 18432, h); }
        { const auto v = hcat({{s + "mlp.down_proj.weight", size_t(h) * 9216}}); L.Wd = tiled(v.data(), h, 9216); }
        L.nin = up16(s + "input_layernorm.weight", h);
        L.npost = up16(s + "post_attention_layernorm.weight", h);
        L.qn = up16(s + "self_attn.q_norm.weight", 128);
        L.kn = up16(s + "self_attn.k_norm.weight", 128);
        L.knc = up16(s + "self_attn.k_norm_ctx.weight", 128);
        L.dfn = up16(s + "dflare_norm.weight", h);
        L.kc = dev_.zeros<uint16_t>(size_t(limit_) * 1024);
        L.vc = dev_.zeros<uint16_t>(size_t(limit_) * 1024);
    }
    d_.mask = up16("mask_embed", h);
    d_.norm = up16("norm.weight", h);
    d_.w1 = up16("markov_head.markov_w1.weight", size_t(Vp) * 256);
    d_.rnnW = up16("markov_head.rnn_proj.weight", size_t(768) * 5120);
    d_.rnnB = up16("markov_head.rnn_proj.bias", 768);
    d_.mdfn = up16("markov_head.dflare_norm.weight", h);
    {   // fuse alpha = softmax(fuse_weight, -1) rounded to BF16
        auto [p, nb] = H("fuse_weight", 32);
        float a[16];
        for (int r = 0; r < 4; ++r) {
            float w[4], mx = -INFINITY, sum = 0.f;
            for (int c = 0; c < 4; ++c) { uint16_t u; std::memcpy(&u, p + 2 * (r * 4 + c), 2); w[c] = b70::bits_to_f32(uint32_t(u) << 16); mx = std::max(mx, w[c]); }
            for (int c = 0; c < 4; ++c) { w[c] = std::exp(w[c] - mx); sum += w[c]; }
            for (int c = 0; c < 4; ++c) a[r * 4 + c] = b70::bf16_to_f32(b70::f32_to_bf16(w[c] / sum));
        }
        d_.alpha = dev_.alloc<float>(16);
        q_.memcpy(d_.alpha, a, sizeof(a));
        std::printf("mtp fuse alpha:");
        for (float v : a) std::printf(" %.4f", v);
        std::printf("\n");
    }
    {   // fold: foldW[V][256] = unembed[V][h] . hidden_out[h][256]; foldB = unembed . hidden_out.bias
        auto [pw, nbw] = H("markov_head.hidden_out.weight", size_t(h) * 256 * 2);
        auto [pb, nbb] = H("markov_head.hidden_out.bias", size_t(h) * 2);
        std::vector<uint16_t> hoT(size_t(256) * h);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(pw);
        for (int i = 0; i < h; ++i) for (int r = 0; r < 256; ++r) hoT[size_t(r) * h + i] = src[size_t(i) * 256 + r];
        uint16_t* dhoT = dev_.alloc<uint16_t>(hoT.size());
        q_.memcpy(dhoT, hoT.data(), hoT.size() * 2).wait();
        float* tmp = sycl::malloc_device<float>(size_t(V) * 256, q_);
        if (!tmp) throw std::runtime_error("mtp fold scratch");
        b70::QuantWeight w;
        w.fmt = b70::Fmt::BF16; w.N = 256; w.K = h; w.payload = reinterpret_cast<const uint8_t*>(dhoT); w.row_bytes = int64_t(h) * 2;
        b70::launch_gemm_fast(q_, w, reinterpret_cast<const bf16x*>(Urow), tmp, V);
        d_.foldW = dev_.alloc<uint16_t>(size_t(V) * 256);
        uint16_t* fw = d_.foldW;
        q_.parallel_for<InkCopy>(sycl::range<1>(size_t(V) * 256), [=](sycl::id<1> id) { fw[id[0]] = tobf(tmp[id[0]]); });
        std::vector<float> hb(h);
        for (int i = 0; i < h; ++i) { uint16_t u; std::memcpy(&u, pb + 2 * i, 2); hb[i] = b70::bits_to_f32(uint32_t(u) << 16); }
        float* dhb = dev_.alloc<float>(h);
        q_.memcpy(dhb, hb.data(), size_t(h) * 4).wait();
        d_.foldB = dev_.alloc<float>(V);
        gemv<0>(Urow, V, h, dhb, d_.foldB, V, nullptr, nullptr, V, nullptr, nullptr);
        q_.wait();
        sycl::free(tmp, q_);
        if (Ufull) sycl::free(Ufull, q_);
        if (i8v_ && !std::getenv("INK_DRAFT_OLD")) {
            // INT8 drafter head (same rounding as the target's INT8 unembed): base logits through b70::launch_fp8_smallm,
            // fold rows int8 with one scale a row
            if (world_ == 1) { d_.UR = unR_; d_.US = unembedS_; }
            else {
                int8_t* qt = nullptr;
                quant_tiled(unembedFullT_, Vp, h, qt, d_.US);
                d_.UR = rowq(qt, Vp, h, kp_);
            }
            d_.foldQ = dev_.alloc<int8_t>(size_t(V) * 256);
            d_.foldS = dev_.alloc<float>(V);
            int8_t* fq = d_.foldQ;
            float* fs = d_.foldS;
            q_.parallel_for<InkFoldQ>(sycl::nd_range<1>(size_t(V) * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const size_t rr = it.get_group(0);
                const int lane = int(sg.get_local_linear_id());
                float mx = 0.f;
                for (int e = 0; e < 16; ++e) mx = sycl::fmax(mx, sycl::fabs(bfw(fw[rr * 256 + lane * 16 + e])));
                mx = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                const float sc = mx > 0.f ? mx / 127.f : 1.f;
                for (int e = 0; e < 16; ++e) fq[rr * 256 + lane * 16 + e] = int8_t(sycl::clamp(sycl::rint(bfw(fw[rr * 256 + lane * 16 + e]) / sc), -127.f, 127.f));
                if (lane == 0) fs[rr] = sc;
            });
            d_.baseR = dev_.zeros<float>(8 * size_t(Vp));
            d_.finb = dev_.zeros<uint16_t>(8 * size_t(kp_));
            q_.wait();
        }
    }
    const size_t mm = size_t(mmax_);
    d_.r = dev_.zeros<float>(8 * size_t(h)); d_.x = dev_.zeros<float>(8 * size_t(h)); d_.qkv = dev_.zeros<float>(8 * 5120);
    d_.att = dev_.zeros<float>(8 * 3072); d_.o = dev_.zeros<float>(8 * size_t(h)); d_.gu = dev_.zeros<float>(8 * 18432);
    d_.act = dev_.zeros<float>(8 * 9216); d_.fin = dev_.zeros<float>(8 * size_t(h)); d_.base = dev_.zeros<float>(8 * size_t(V));
    d_.state = dev_.zeros<float>(256); d_.inp = dev_.zeros<float>(5120); d_.proj = dev_.zeros<float>(768); d_.latent = dev_.zeros<float>(256);
    d_.dlog = dev_.zeros<float>(V); d_.frnn = dev_.zeros<float>(h);
    d_.ctx = dev_.zeros<float>(mm * h); d_.ctxb = dev_.zeros<uint16_t>(mm * h); d_.kvc = dev_.zeros<float>(mm * 2048);
    d_.bk = dev_.zeros<uint16_t>(8 * 1024); d_.bv = dev_.zeros<uint16_t>(8 * 1024); d_.dtok = dev_.zeros<int>(8);
    q_.wait();
    d_.st = st;
    mtp_ = true;
    taps_on = true;
    std::printf("mtp drafter loaded (3 layers, block 8, markov rank 256); device total %.2f GiB\n", dev_.bytes / 1073741824.0);
}

// Per-head RMSNorm (128) + NeoX RoPE (theta 1e6) for one row of q or k; one sub-group per head.
static inline void head_norm_rope(sycl::sub_group sg, const float* in, float* outf, uint16_t* outb, const uint16_t* w, int pos, float eps) {
    const int lane = int(sg.get_local_linear_id());
    float v[8], ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) { v[e] = in[lane * 8 + e]; ss += v[e] * v[e]; }
    ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
    const float r = 1.f / sycl::sqrt(ss / 128.f + eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = bfr((v[e] * r) * bfw(w[lane * 8 + e]));
    // lanes 0..7 hold dims 0..63, lanes 8..15 dims 64..127: partner of dim d is lane ^ 8
    float o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float partner = sycl::permute_group_by_xor(sg, v[e], 8);
        const int d = (lane & 7) * 8 + e;
        const float inv = sycl::pow(1000000.f, -float(2 * d) / 128.f);
        const float ang = float(pos) * inv;
        const float cs = sycl::cos(ang), sn = sycl::sin(ang);
        o[e] = lane < 8 ? bfr(v[e] * cs - partner * sn) : bfr(v[e] * cs + partner * sn);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        if (outf) outf[lane * 8 + e] = o[e];
        if (outb) outb[lane * 8 + e] = tobf(o[e]);
    }
}

// Mix the four target taps with alpha[row], RMSNorm with w; rows of out (float, BF16-rounded) + BF16 copy.
void Engine::mix_norm_taps(int alpha_row, const uint16_t* w, int m0, int rows, float* out, uint16_t* outb) {
    const float* pool = pool_;
    const float* alpha = d_.alpha + alpha_row * 4;
    const int h = h_;
    const size_t RS = rs_;
    const float eps = c_.eps;
    q_.parallel_for<InkMixNorm>(sycl::nd_range<1>(size_t(rows) * 256, 256), [=](sycl::nd_item<1> it) {
        const int m = m0 + int(it.get_group(0)), lid = int(it.get_local_id(0));
        float ss = 0.f;
        for (int i = lid; i < h; i += 256) {
            float v = 0.f;
            for (int t = 0; t < 4; ++t) v += bfr(pool[size_t(TAP0 + t) * RS + size_t(m) * h + i]) * alpha[t];
            ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.f / sycl::sqrt(ss / float(h) + eps);
        const size_t o = size_t(m - m0) * h;
        for (int i = lid; i < h; i += 256) {
            float v = 0.f;
            for (int t = 0; t < 4; ++t) v += bfr(pool[size_t(TAP0 + t) * RS + size_t(m) * h + i]) * alpha[t];
            const float y = bfr(v * r * bfw(w[i]));
            out[o + i] = y;
            if (outb) outb[o + i] = tobf(y);
        }
    });
}

// Context K/V of the drafter for target rows m0.. at positions p0.. ; f_rnn from the last row.
void Engine::draft_commit(int p0, int m0, int rows) {
    const int h = h_;
    static bool dumped = false;
    const bool dump = !dumped && rows >= 32 && std::getenv("INK_DUMP");
    if (dump)
        for (int tp = 0; tp < 4; ++tp) dump_dev(q_, ("taps" + std::to_string(tp)).c_str(), row(TAP0 + tp, m0), size_t(rows) * h * 4);
    for (int l = 0; l < 3; ++l) {
        DLayer& L = d_.L[l];
        mix_norm_taps(l, L.dfn, m0, rows, d_.ctx, d_.ctxb);
        for (int r0 = 0; r0 < rows; r0 += (rows >= 32 ? rows : 8)) {
            const int n = rows >= 32 ? rows : std::min(8, rows - r0);
            dense(L.Wkvc, 2048, h, d_.ctx + size_t(r0) * h, h, rows >= 32 ? d_.ctxb : nullptr, d_.kvc + size_t(r0) * 2048, 2048, n, true);
        }
        const float* kv = d_.kvc;
        uint16_t *kc = L.kc, *vc = L.vc;
        const uint16_t* kn = L.knc;
        const float eps = c_.eps;
        q_.parallel_for<InkKVCtx>(sycl::nd_range<1>(size_t(rows) * 16 * 16, 16 * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            const int m = int(it.get_group(0)), hd = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
            const int pos = p0 + m;
            const float* row = kv + size_t(m) * 2048;
            if (hd < 8) head_norm_rope(sg, row + hd * 128, nullptr, kc + size_t(pos) * 1024 + hd * 128, kn, pos, eps);
            else {
                const int vh = hd - 8;
#pragma unroll
                for (int e = 0; e < 8; ++e) vc[size_t(pos) * 1024 + vh * 128 + lane * 8 + e] = tobf(row[1024 + vh * 128 + lane * 8 + e]);
            }
        });
    }
    mix_norm_taps(3, d_.mdfn, m0 + rows - 1, 1, d_.frnn, nullptr);
    if (dump) {
        dumped = true;
        for (int l = 0; l < 3; ++l) {
            dump_dev(q_, ("kc" + std::to_string(l)).c_str(), d_.L[l].kc + size_t(p0) * 1024, size_t(rows) * 1024 * 2);
            dump_dev(q_, ("vc" + std::to_string(l)).c_str(), d_.L[l].vc + size_t(p0) * 1024, size_t(rows) * 1024 * 2);
        }
        dump_dev(q_, "frnn", d_.frnn, size_t(h) * 4);
        std::printf("dumped prompt taps + context K/V (%d rows at %d)\n", rows, p0);
    }
}

// One draft block at positions p..p+7: row 0 = target embedding of t, rows 1..7 = mask_embed.
// Writes drafts to d_.dtok[1..7] (d_.dtok[0] = t).
void Engine::draft_block(int t, int p_in) {
    static const int pshift = std::getenv("INK_POSSHIFT") ? std::atoi(std::getenv("INK_POSSHIFT")) : 0;
    const int p = p_in + pshift;
    const int h = h_, V = c_.output_vocab;
    const float eps = c_.eps;
    float* r = d_.r;
    {
        const uint16_t *Em = embed_, *wn = embed_norm_, *mask = d_.mask;
        int* dt = d_.dtok;
        q_.parallel_for<InkDEmbed>(sycl::nd_range<1>(8 * 256, 256), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            if (m == 0) {
                const uint16_t* e = Em + size_t(t) * h;
                float ss = 0.f;
                for (int i = lid; i < h; i += 256) { const float v = bfw(e[i]); ss += v * v; }
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                const float rr = 1.f / sycl::sqrt(ss / float(h) + eps);
                for (int i = lid; i < h; i += 256) r[i] = bfr((bfw(e[i]) * rr) * bfw(wn[i]));
                if (lid == 0) dt[0] = t;
            } else
                for (int i = lid; i < h; i += 256) r[size_t(m) * h + i] = bfw(mask[i]);
        });
    }
    auto norm8 = [&](const float* in, const uint16_t* w, float* out, int rows, int m0) {
        q_.parallel_for<InkDNorm>(sycl::nd_range<1>(size_t(rows) * 256, 256), [=](sycl::nd_item<1> it) {
            const int m = m0 + int(it.get_group(0)), lid = int(it.get_local_id(0));
            const float* x = in + size_t(m) * h;
            float ss = 0.f;
            for (int i = lid; i < h; i += 256) ss += x[i] * x[i];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float rr = 1.f / sycl::sqrt(ss / float(h) + eps);
            for (int i = lid; i < h; i += 256) out[size_t(m) * h + i] = bfr((x[i] * rr) * bfw(w[i]));
        });
    };
    auto addres = [&](const float* add) {
        q_.parallel_for<InkAddRes>(sycl::range<1>(8 * size_t(h)), [=](sycl::id<1> id) { r[id[0]] = bfr(r[id[0]] + add[id[0]]); });
    };
    static bool bdumped = false;
    const bool bd = !bdumped && std::getenv("INK_DUMP");
    if (bd) { dump_dev(q_, "r_in", r, 8 * size_t(h) * 4); int pp = p; FILE* f = std::fopen((std::string(std::getenv("INK_DUMP")) + "/meta.txt").c_str(), "w"); std::fprintf(f, "%d %d\n", t, pp); std::fclose(f); }
    for (int l = 0; l < 3; ++l) {
        DLayer& L = d_.L[l];
        norm8(r, L.nin, d_.x, 8, 0);
        gemvtm<0>(L.Wqkv, 5120, h, 4, 8, d_.x, d_.x, d_.x, h, d_.qkv, 5120, 5120, nullptr, 0, 5120, nullptr, 0, 5120);
        {
            float* qkv = d_.qkv;
            uint16_t *bk = d_.bk, *bv = d_.bv;
            const uint16_t *qn = L.qn, *kn = L.kn;
            q_.parallel_for<InkQKRope>(sycl::nd_range<1>(size_t(8) * 40 * 16, 40 * 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int m = int(it.get_group(0)), hd = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                float* row = qkv + size_t(m) * 5120;
                if (hd < 24) head_norm_rope(sg, row + hd * 128, row + hd * 128, nullptr, qn, p + m, eps);
                else if (hd < 32) head_norm_rope(sg, row + 3072 + (hd - 24) * 128, nullptr, bk + size_t(m) * 1024 + (hd - 24) * 128, kn, p + m, eps);
                else {
#pragma unroll
                    for (int e = 0; e < 8; ++e) bv[size_t(m) * 1024 + (hd - 32) * 128 + lane * 8 + e] = tobf(bfr(row[4096 + (hd - 32) * 128 + lane * 8 + e]));
                }
            });
        }
        {   // attention: context [lo, p) + the 8 block rows, non-causal; one sub-group per (row, head), lane per key
            const float* qkv = d_.qkv;
            float* att = d_.att;
            const uint16_t *kc = L.kc, *vc = L.vc, *bk = d_.bk, *bv = d_.bv;
            static const bool noctx = std::getenv("INK_NOCTX") != nullptr;
            const int lo = noctx ? p_in : L.sliding ? std::max(0, p_in - 1023) : 0, nctx = p_in - lo, count = nctx + 8;
            const float scale = 1.f / std::sqrt(128.f);
            q_.submit([&](sycl::handler& cgh) {
                sycl::local_accessor<float, 1> qsm(sycl::range<1>(8 * 128), cgh);
                cgh.parallel_for<InkDAtt>(sycl::nd_range<1>(size_t(8 * 24 / 8) * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg = it.get_sub_group();
                    const int sgid = int(sg.get_group_linear_id()), lane = int(sg.get_local_linear_id());
                    const int t2 = int(it.get_group(0)) * 8 + sgid, m = t2 / 24, j = t2 % 24, kvh = j / 3;
                    float* qq = qsm.template get_multi_ptr<sycl::access::decorated::no>().get() + sgid * 128;
#pragma unroll
                    for (int e = 0; e < 8; ++e) qq[lane * 8 + e] = qkv[size_t(m) * 5120 + j * 128 + lane * 8 + e];
                    sycl::group_barrier(sg);
                    auto krow = [&](int ii) { return ii < nctx ? kc + size_t(lo + ii) * 1024 + kvh * 128 : bk + size_t(ii - nctx) * 1024 + kvh * 128; };
                    auto vrow = [&](int ii) { return ii < nctx ? vc + size_t(lo + ii) * 1024 + kvh * 128 : bv + size_t(ii - nctx) * 1024 + kvh * 128; };
                    auto score = [&](int ii) {
                        const uint16_t* kr = krow(ii);
                        float s = 0.f;
                        for (int d0 = 0; d0 < 128; d0 += 8) {
                            const sycl::vec<uint16_t, 8> kv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(kr + d0);
#pragma unroll
                            for (int e = 0; e < 8; ++e) s += qq[d0 + e] * bfw(kv[e]);
                        }
                        return s * scale;
                    };
                    float mx = -INFINITY, lsum = 0.f;
                    for (int ii = lane; ii < count; ii += 16) {
                        const float s = score(ii);
                        if (s > mx) { lsum = lsum * sycl::exp(mx - s) + 1.f; mx = s; } else lsum += sycl::exp(s - mx);
                    }
                    const float gm = sycl::reduce_over_group(sg, mx, sycl::maximum<float>());
                    const float sum = sycl::reduce_over_group(sg, mx == -INFINITY ? 0.f : lsum * sycl::exp(mx - gm), sycl::plus<float>());
                    float acc[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
                    for (int base = 0; base < count; base += 16) {
                        const int ii = base + lane;
                        const float w = ii < count ? sycl::exp(score(ii) - gm) / sum : 0.f;
                        const int nk = sycl::min(16, count - base);
                        for (int kk = 0; kk < nk; ++kk) {
                            const float wk = sycl::select_from_group(sg, w, kk);
                            const sycl::vec<uint16_t, 8> vv = *reinterpret_cast<const sycl::vec<uint16_t, 8>*>(vrow(base + kk) + lane * 8);
#pragma unroll
                            for (int e = 0; e < 8; ++e) acc[e] += wk * bfw(vv[e]);
                        }
                    }
#pragma unroll
                    for (int e = 0; e < 8; ++e) att[size_t(m) * 3072 + j * 128 + lane * 8 + e] = bfr(acc[e]);
                });
            });
        }
        if (bd) { dump_dev(q_, ("qkv" + std::to_string(l)).c_str(), d_.qkv, 8 * 5120 * 4); dump_dev(q_, ("att" + std::to_string(l)).c_str(), d_.att, 8 * 3072 * 4); }
        gemvtm<0>(L.Wo, h, 3072, 8, 8, d_.att, d_.att, d_.att, 3072, d_.o, h, h, nullptr, 0, h, nullptr, 0, h);
        addres(d_.o);
        if (bd) dump_dev(q_, ("r_attn" + std::to_string(l)).c_str(), r, 8 * size_t(h) * 4);
        norm8(r, L.npost, d_.x, 8, 0);
        gemvtm<0>(L.Wgu, 18432, h, 1, 8, d_.x, d_.x, d_.x, h, d_.gu, 18432, 18432, nullptr, 0, 18432, nullptr, 0, 18432);
        {
            const float* gu = d_.gu;
            float* act = d_.act;
            q_.parallel_for<InkSilu>(sycl::range<1>(8 * 9216), [=](sycl::id<1> id) {
                const size_t m = id[0] / 9216, i = id[0] % 9216;
                const float g = gu[m * 18432 + i], u = gu[m * 18432 + 9216 + i];
                act[id[0]] = bfr(bfr(g / (1.f + sycl::exp(-g))) * u);
            });
        }
        gemvtm<0>(L.Wd, h, 9216, 8, 8, d_.act, d_.act, d_.act, 9216, d_.o, h, h, nullptr, 0, h, nullptr, 0, h);
        addres(d_.o);
        if (bd) dump_dev(q_, ("r_mlp" + std::to_string(l)).c_str(), r, 8 * size_t(h) * 4);
    }
    norm8(r, d_.norm, d_.fin, 7, 1);
    const bool i8head = d_.UR != nullptr;
    if (i8head) {
        uint16_t* fb = d_.finb;
        const float* fin = d_.fin;
        const int kp = kp_;
        q_.parallel_for<InkFinB>(sycl::range<1>(size_t(7) * h), [=](sycl::id<1> id) { const size_t m = id[0] / h, i = id[0] % h; fb[(1 + m) * kp + i] = tobf(fin[(1 + m) * h + i]); });
        if (!i8_small(d_.UR, d_.US, 0, c_.vocab, kp_, d_.finb + kp_, d_.baseR, 7)) throw std::runtime_error("drafter: INT8 head refused");
    } else
    gemvtm<0>(unembedFullT_, c_.vocab, h, 1, 7, d_.fin + h, d_.fin + h, d_.fin + h, h, d_.base, V, c_.vocab, nullptr, 0, c_.vocab, nullptr, 0, V);
    if (bd) { dump_dev(q_, "fin", d_.fin, 8 * size_t(h) * 4); dump_dev(q_, "base", d_.base, 7 * size_t(V) * 4); }
    // Markov walk: 7 sequential steps on the device
    q_.memset(d_.state, 0, 256 * sizeof(float));
    for (int step = 0; step < 7; ++step) {
        {
            float* inp = d_.inp;
            const float *st = d_.state, *fin = d_.fin + size_t(1 + step) * h, *fr = d_.frnn;
            const uint16_t* w1 = d_.w1;
            const int* dt = d_.dtok;
            q_.parallel_for<InkRnnIn>(sycl::range<1>(5120), [=](sycl::id<1> id) {
                const int i = int(id[0]);
                const int prev = dt[step];
                inp[i] = i < 256 ? st[i] : i < 512 ? bfw(w1[size_t(prev) * 256 + (i - 256)]) : i < 512 + h ? fin[i - 512] : fr[i - 512 - h];
            });
        }
        gemv<1>(d_.rnnW, 768, 5120, d_.inp, d_.proj, 768, nullptr, nullptr, 768, nullptr, nullptr);
        {
            const float* pr = d_.proj;
            const uint16_t* rb = d_.rnnB;
            float *st = d_.state, *lat = d_.latent;
            q_.parallel_for<InkRnn>(sycl::range<1>(256), [=](sycl::id<1> id) {
                const int i = int(id[0]);
                const float gr = bfr(pr[i] + bfw(rb[i])), cr = bfr(pr[256 + i] + bfw(rb[256 + i])), orw = bfr(pr[512 + i] + bfw(rb[512 + i]));
                const float gate = bfr(1.f / (1.f + sycl::exp(-gr)));
                const float kept = bfr(gate * st[i]);
                const float blend = bfr(bfr(1.f - gate) * bfr(sycl::tanh(cr)));
                st[i] = bfr(kept + blend);
                lat[i] = bfr(sycl::tanh(orw));
            });
        }
        {   // logits = softcap(bf(base + bf(foldB + foldW . latent))), argmax over the unpadded vocab
            const uint16_t* fw = d_.foldW;
            const float *fb = d_.foldB, *lat = d_.latent, *base = i8head ? d_.baseR + size_t(step) * c_.vocab : d_.base + size_t(step) * V;
            const int8_t* fq = d_.foldQ;
            const float* fsc = d_.foldS;
            float* lg = d_.dlog;
            const float cap = c_.logit_cap;
            static const float bias_on = std::getenv("INK_NOBIAS") ? 0.f : 1.f;
            const float bon = bias_on;
            const size_t groups = (size_t(V) + 7) / 8;
            q_.parallel_for<InkFold>(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                auto sg = it.get_sub_group();
                const int rr = int(it.get_group(0)) * 8 + int(sg.get_group_linear_id());
                if (rr >= V) return;
                const int lane = int(sg.get_local_linear_id());
                float s = 0.f;
                if (fq) {
                    const sycl::vec<int8_t, 16> qv = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(fq + size_t(rr) * 256 + lane * 16);
#pragma unroll
                    for (int e = 0; e < 16; ++e) s += float(qv[e]) * lat[lane * 16 + e];
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>()) * fsc[rr];
                } else {
                    const sycl::vec<uint16_t, 16> wv = *reinterpret_cast<const sycl::vec<uint16_t, 16>*>(fw + size_t(rr) * 256 + lane * 16);
#pragma unroll
                    for (int e = 0; e < 16; ++e) s += bfw(wv[e]) * lat[lane * 16 + e];
                    s = sycl::reduce_over_group(sg, s, sycl::plus<float>());
                }
                if (lane == 0) {
                    const float bias = bon * bfr(fb[rr] + s);
                    const float v = bfr(bfr(base[rr]) + bias);
                    lg[rr] = cap * sycl::tanh(v / cap);
                }
            });
        }
        if (i8head) argmax_rows_cap(d_.dlog, V, V, 1, d_.dtok + 1 + step, nullptr, 0, 0.f);
        else argmax_rows(d_.dlog, V, 1, d_.dtok + 1 + step);
    }
    if (bd) { bdumped = true; dump_dev(q_, "dtok", d_.dtok, 8 * 4); std::printf("dumped first draft block\n"); }
}

// Row-wise argmax (lowest index on ties) of rows x [n][V] into out[n].
void Engine::argmax_rows(const float* lg, int V, int n, int* out) {
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lv(sycl::range<1>(1024), cgh);
        sycl::local_accessor<int, 1> li(sycl::range<1>(1024), cgh);
        cgh.parallel_for<InkArgM1>(sycl::nd_range<1>(size_t(n) * 1024, 1024), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            const float* x = lg + size_t(m) * V;
            float best = -INFINITY; int bidx = 0x7fffffff;
            for (int k = lid; k < V; k += 1024) if (x[k] > best) { best = x[k]; bidx = k; }
            lv[lid] = best; li[lid] = bidx;
            for (int s = 512; s > 0; s >>= 1) {
                sycl::group_barrier(it.get_group());
                if (lid < s && (lv[lid + s] > lv[lid] || (lv[lid + s] == lv[lid] && li[lid + s] < li[lid]))) { lv[lid] = lv[lid + s]; li[lid] = li[lid + s]; }
            }
            if (lid == 0) out[m] = li[0];
        });
    });
}

// Raw logits (ld-pitched rows, first V valid): value = cap tanh(bf(raw) / cap) (the gemvtm<2> formula), argmax with the
// lowest index on ties; out = base + index, val (optional) = the value.  Two stages: 4096 logits a work-group, then rows.
void Engine::argmax_rows_cap(const float* lg, int ld, int V, int n, int* out, float* val, int base, float cap) {
    const int per = 4096, G = (V + per - 1) / per;
    if (G > 64) throw std::runtime_error("argmax_rows_cap: vocabulary too wide");
    float* pv = amxm_v_;
    int* pi = amxm_i_;
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lv(sycl::range<1>(256), cgh);
        sycl::local_accessor<int, 1> li(sycl::range<1>(256), cgh);
        cgh.parallel_for<InkArgC1>(sycl::nd_range<1>(size_t(n) * G * 256, 256), [=](sycl::nd_item<1> it) {
            const int grp = int(it.get_group(0)), m = grp / G, gb = grp % G, lid = int(it.get_local_id(0));
            const float* x = lg + size_t(m) * ld;
            float best = -INFINITY; int bidx = 0x7fffffff;
            for (int t = 0; t < 16; ++t) {
                const int k = gb * per + t * 256 + lid;
                if (k >= V) continue;
                const float v = cap > 0.f ? cap * sycl::tanh(bfr(x[k]) / cap) : x[k];
                if (v > best || (v == best && k < bidx)) { best = v; bidx = k; }
            }
            lv[lid] = best; li[lid] = bidx;
            for (int s2 = 128; s2 > 0; s2 >>= 1) {
                sycl::group_barrier(it.get_group());
                if (lid < s2 && (lv[lid + s2] > lv[lid] || (lv[lid + s2] == lv[lid] && li[lid + s2] < li[lid]))) { lv[lid] = lv[lid + s2]; li[lid] = li[lid + s2]; }
            }
            if (lid == 0) { pv[m * 64 + gb] = lv[0]; pi[m * 64 + gb] = li[0]; }
        });
    });
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lv(sycl::range<1>(64), cgh);
        sycl::local_accessor<int, 1> li(sycl::range<1>(64), cgh);
        cgh.parallel_for<InkArgC2>(sycl::nd_range<1>(size_t(n) * 64, 64), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            lv[lid] = lid < G ? pv[m * 64 + lid] : -INFINITY;
            li[lid] = lid < G ? pi[m * 64 + lid] : 0x7fffffff;
            for (int s2 = 32; s2 > 0; s2 >>= 1) {
                sycl::group_barrier(it.get_group());
                if (lid < s2 && (lv[lid + s2] > lv[lid] || (lv[lid + s2] == lv[lid] && li[lid + s2] < li[lid]))) { lv[lid] = lv[lid + s2]; li[lid] = li[lid + s2]; }
            }
            if (lid == 0) { out[m] = li[0] + base; if (val) val[m] = lv[0]; }
        });
    });
}

// Row-wise argmax over the first V of ld-pitched rows: global index (base + i) and value (lowest index on ties).
void Engine::argmax_rows_v(const float* lg, int ld, int V, int n, int* out, float* val, int base) {
    q_.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> lv(sycl::range<1>(1024), cgh);
        sycl::local_accessor<int, 1> li(sycl::range<1>(1024), cgh);
        cgh.parallel_for<InkArgMV>(sycl::nd_range<1>(size_t(n) * 1024, 1024), [=](sycl::nd_item<1> it) {
            const int m = int(it.get_group(0)), lid = int(it.get_local_id(0));
            const float* x = lg + size_t(m) * ld;
            float best = -INFINITY; int bidx = 0x7fffffff;
            for (int k = lid; k < V; k += 1024) if (x[k] > best) { best = x[k]; bidx = k; }
            lv[lid] = best; li[lid] = bidx;
            for (int s = 512; s > 0; s >>= 1) {
                sycl::group_barrier(it.get_group());
                if (lid < s && (lv[lid + s] > lv[lid] || (lv[lid + s] == lv[lid] && li[lid + s] < li[lid]))) { lv[lid] = lv[lid + s]; li[lid] = li[lid + s]; }
            }
            if (lid == 0) { out[m] = li[0] + base; val[m] = lv[0]; }
        });
    });
}

// One speculative step: draft from (t, pos), verify [t, d1..d7], keep the accepted prefix.
// Returns the emitted tokens (accepted drafts + the target's next token).
std::vector<int> Engine::mtp_step(int t, const int* forced) {
    const int p = pos_;
    if (p + 8 > limit_) throw std::runtime_error("context full");
    static const bool mtime = std::getenv("INK_MTP_TIME") != nullptr;   // per-phase host wall times (adds waits)
    auto tA = std::chrono::steady_clock::now();
    if (forced) {
        int dt[8] = {t};
        for (int k = 0; k < 7; ++k) dt[1 + k] = forced[k];
        q_.memcpy(d_.dtok, dt, sizeof(dt));
    } else draft_block(t, p);
    if (mtime) { q_.wait(); const auto t = std::chrono::steady_clock::now(); mtp_ms_[0] += std::chrono::duration<double, std::milli>(t - tA).count(); tA = t; }
    q_.memcpy(tok_, d_.dtok, 8 * sizeof(int));
    verify_ = all_logits_ = true;
    run(8);
    verify_ = all_logits_ = false;
    int nx[8], dr[8];
    q_.memcpy(dr, d_.dtok, 8 * sizeof(int));
    final_tokens(8, nx);
    if (mtime) { q_.wait(); const auto t = std::chrono::steady_clock::now(); mtp_ms_[1] += std::chrono::duration<double, std::milli>(t - tA).count(); tA = t; }
    int a = 0;
    while (a < 7 && nx[a] == dr[a + 1]) ++a;
    static int dbg = std::getenv("INK_DEBUG") ? std::atoi(std::getenv("INK_DEBUG")) : 0;
    if (dbg > 0) {
        --dbg;
        std::printf("[verify p=%d] drafts:", p);
        for (int k = 0; k < 8; ++k) std::printf(" %d", dr[k]);
        std::printf("\n[verify p=%d] target:", p);
        for (int k = 0; k < 8; ++k) std::printf(" %d", nx[k]);
        std::printf("  accepted %d\n", a);
    }
    commit_verify(p, a);
    draft_commit(p, 0, a + 1);
    pos_ = p + a + 1;
    q_.memcpy(pos_d_, &pos_, sizeof(int));
    if (mtime) { q_.wait(); mtp_ms_[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tA).count(); }
    std::vector<int> out(dr + 1, dr + 1 + a);
    out.push_back(nx[a]);
    return out;
}
