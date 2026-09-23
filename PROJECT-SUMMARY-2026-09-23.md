# GRIMOIRE — Project Summary

**Generated:** 2026-09-23 · **Branch:** `main` (`389df07`) · **Span:** 2026-08-28 → 2026-09-23 (27 days)
**Scale:** 256 commits · 167 files touched · +58,647 / −1,520 lines · 39 engine source files · 130 tools/scripts · 19 host test files · 18 device-gate binaries

GRIMOIRE is a from-scratch C++/SYCL inference engine for Intel Arc Pro B70 (Battlemage) GPUs — no PyTorch, no vLLM at runtime. This table is the full chronological record of what was built, fixed, measured, and audited, drawn from the git history and the 46 dated `.md` handoffs in the repo.

**How to read the Status column:** ✅ Verified = has a passing automated gate (a `bin/test_*` binary) or was measured with real generated text (project rule 8: "never trust a number that didn't generate text"). 🖥️ CPU-only = correct on the CPU SYCL fallback, never yet run on a B70. ⏳ Open = known gap, not yet built or not yet measured. ⚠️ Reverted = tried and rolled back.

---

## 1. Timeline of work

| Dates | Phase | What was done | Status |
|---|---|---|---|
| **08-28** | Initial commit & scaffolding | From-scratch engine committed; `CLAUDE.md` conventions written (freed-payload rule, DAG-race rule, stale-bridge rule); original z-lab DFlash drafter path started for Ornith-1.5-35B-A3B (context ingestion, 16-query block forward) | 🖥️ CPU-only |
| **08-28 → 08-29** | Muse Glimmer DFlash | Coherent DFlash path for Muse; PP baseline probe found a host-side SIGSEGV on MXFP4 prefill; real baseline measured (~602 tok/s, 3.5× below roofline); verifier dispatch optimized, narrow-tile verifier added, feature projection accelerated | ✅ Verified (device, pre-refactor) |
| **08-30 → 08-31** | Muse DFlash parity chase | INT4 compressed loading; a "Fusion V2" runtime path added then **reverted**; native vs. vLLM Muse verifier compared tensor-for-tensor; fixed an attention-bridge page-stride bug (paged reads past page 0) and an FP16 scratch overflow — **speculation went from 0/8 to 9/11 accepted**; found the real prefill gap was dispatch/fusion overhead, not the GEMMs themselves | ✅ Verified; ⚠️ 1 revert |
| **09-01 → 09-03** | Qwen MTP + Ornith DFlash2, HTTP server | Qwen W4A8+MTP: TG 31.0→37.4, PP→2499; stale int8 activation-cache bug fixed (0%→69% MTP acceptance); PP restored to 2540/TG 49.8. Ornith: DFlash2 block drafting tried, reverted, retried — reached 97.6 TG but **lost to plain 125 TG decode**; multiple drafter candidates evaluated (NInfer, shisa-ai) and rejected in favor of the native head. `grimoire-server` built: OpenAI-compatible SSE streaming, speculative decode via `--dflash-model`, Harmony reasoning-stream handling. First head-to-head: **GRIMOIRE speculation beats vLLM**; the remaining gap traced to vLLM's FlashAttention-2 vs. GRIMOIRE's plain forward | ✅ Verified; ⚠️ 2 reverts |
| **09-04 → 09-05** | Decode-path deep optimization | Found server never builds its own decode graph (documented, later wired — see 09-22). Attention split-K, batched-attention M=1 failure narrowed to workgroup geometry. VRAM bandwidth measured directly: 625.7 GB/s raw, 586.8 GB/s trustworthy — proved the GEMV *memory-access pattern*, not the quant format, was the bottleneck (dequant itself costs only 1.6%). Routed MXFP4 and W4A8 GEMVs through oneDNN: **base decode TG 23.25 → ~30**. Fixed a UTF-8 tokenizer bug that was byte-fallback-corrupting real prose (7475→5983 tokens on the same text) and a `\uXXXX` JSON-decoding bug corrupting every non-ASCII prompt | ✅ Verified (device) |
| **09-06 → 09-11** | K2-Horizon + format work | Routed speculative verify through W4A16: **tg32 30.11 → 44.36**. Ornith MoE dispatch aligned with vLLM; GPTQ model identity pinned; native INT4/FP8 export with round-trip reload validation added. Started K2-Horizon architecture support: config, grouped RMSNorm, softplus gate, sigmoid router, MoVA tensor mapping and engine wiring | ✅ Verified (device) |
| **09-12** (33 commits) | Toolchain + K2 + rules audit | **Set up a real SYCL compiler in a work container** (no GPU needed) — immediately found `grimoire.cpp` didn't even compile, a K2 softplus-gate precision bug, and 4 K2 engine bugs including a 46 KB device heap overrun that is a `DEVICE_LOST` on real hardware. Fixed a TP scratch-buffer overrun. Built `preflight_b70.sh`. Wrote a capability-matrix banner. DFlash2's candidate selector was found to have **never been called** — its earlier "loses to MTP" verdict was retracted as invalid. Hybrid (DeltaNet) speculation covered for the first time; a rollback-corruption bug found and fixed | ✅🖥️ First compiler-verified state |
| **09-13** (24 commits) | DFlash config fidelity, multi-GPU | Drafter `config.json` resolution rewritten to match the reference exactly (was hardcoded rope_theta, head_dim, etc.). DFlash extended to run under **both TP and PP** (was single-GPU only) — two silent bugs fixed (block embed read the wrong sharded table; drafter shared the target's sharded lm_head). Agnes-3.0-Flash architecture recognized, `parallel_ffn` implemented by folding into the main FFN. **Two full external audits** run against the branch; all findings fixed | ✅ Verified (device gates) |
| **09-14 → 09-15** | gemma-4 | Reference extracted and read layer-by-layer (not inferred from siblings — this is what rule 11 in `CLAUDE.md` exists to prevent). Architecture, GeGLU, proportional RoPE, sandwich residual graph, per-layer attention geometry all built and pinned against the reference. **Three external audits** on this one model each found real bugs (norm convention, value path, RoPE factor capture, `tie_word_embeddings` parsed-but-unused causing constant output). head_dim raised to 512 by templating the flash accumulator instead of a blanket bump. Batched prefill added, made default | ✅ Verified (device gates) |
| **09-16 → 09-17** | Qwen4-Exp, concurrency, NVFP4 | Qwen3.8-Flash-Next: HyperConnections, QSA (indexer + sparse attention), PLE n-gram embedding — each proven live by A/B against the same checkpoint with the mechanism switched off. Device-capability detection consolidated to one function (`device_can_matrix`) after a bad CPU aspect flag caused 4 gates to crash simultaneously. NVFP4 (NVIDIA Blackwell) checkpoint reading added. **The server was found to be fully serial** (measured against an AMD RDNA4 stack) — this triggered the whole concurrency rewrite: prefix-cache conversation resume (pointer move, not copy), multi-sequence batched decode, and removing the per-request server mutex | ✅ Verified (device gates) |
| **09-21** | External audit round 3 + PP serving | Server extended to serve across **two cards** over HTTP (was CLI-only). External audit found **9 real defects, 5 P1**, in code that each had its own green gate — including an inverted NVFP4 dequant formula (wrong by the scale²) and two concurrent requests able to silently share one physical KV slot. All 9 fixed and independently re-verified against the audit's own reproductions or the upstream source | ✅ Verified (device gates) |
| **09-22** (composable-serving) | Batching made to compose with everything | 9 commits made batching work **together** with the prefix cache, TP, PP, a loaded MTP/DFlash drafter (one card), and Muse/gemma-4/Qwen4-Exp — all previously mutually exclusive. Fixed a one-token-past-context budget bug found by the gate itself. CPU-SYCL CI workflow added | ✅ Verified (CPU device) |
| **09-22 → 09-23** | Independent audit + hardening (this session) | Verified the branch state, compiled and ran the **full 15-gate + 17-host-suite** off-card, found and fixed: unsafe partial allocations on failure (device-write-through-null risk), TP ranks that could silently disagree on how much of a cached prompt each had reused, an O(rows) allocate/free per TP GEMM call, a shell script that reported success on a crashed server, batched MTP speculation that had *never actually accepted a draft in any test* (an equality gate can't see a branch it never exercises). Added `serve_tp2.sh` (TP had no launcher). Rewrote all docs that described pre-09-22 refusals. Merged everything to `main` | ✅ Verified (CPU device + CI) |

---

## 2. What is verified, and how

| Claim | Verified by | Confidence |
|---|---|---|
| Every SYCL kernel compiles and matches its host reference | `test_k2_kernels` | ✅ Device |
| 10 architectures × 7 quant formats all load and generate real text | `test_model_matrix` (0 failed cells) | ✅ CPU device |
| PP and TP give token-identical output to a single process | `test_parallel_e2e` (50/50 matches) | ✅ CPU device |
| Speculation (MTP + DFlash) is byte-identical to plain greedy decode | `test_spec_e2e` | ✅ CPU device |
| Batched decode == serial decode, per sequence | `test_batch_decode` | ✅ CPU device |
| Conversations resume from a cache instead of re-reading | `test_prefix_reuse`, `test_batch_prefix` | ✅ CPU device |
| Concurrent requests answer what they answer alone | `test_scheduler`, `test_batch_parallel` | ✅ CPU device |
| A resident two-card server serves multiple requests correctly | `test_pp_server` | ✅ CPU device |
| Batched speculation actually accepts drafts (not just runs) | `test_batch_spec` (forced-acceptance head, negative-controlled) | ✅ CPU device |
| Host-side correctness (formats, tokenizer, HTTP, generation template) | 17 `make test` / `make test-correctness` suites | ✅ Host |

**Not yet verified — needs the Tower (2× B70 on OCuLink):**
- Every number claiming speed, throughput, or latency in every doc in this repo (project rule 8) — nothing here was benchmarked, only checked for correctness on a CPU fallback.
- Every XMX/`joint_matrix` kernel — the CPU device cannot execute them at all.
- The OCuLink Gen4×4 link itself, under both PP and TP.
- Real VRAM fit for `GRIMOIRE_SEQ_SLOTS` × (KV cache + recurrent state + drafter copies) at production `--ctx`.
- The AOT (`bmg_g31`) build path and the cutlass bridges (`tools/build_bridges_b70.sh`), which only build inside the real container image.

---

## 3. Architectures supported

| Architecture | Status | Batched decode | Speculation | PP/TP |
|---|---|---|---|---|
| Dense (Qwen-style) | ✅ | ✅ | ✅ MTP + DFlash | ✅ |
| MoE | ✅ | ✅ | ✅ MTP + DFlash | ✅ |
| Hybrid / DeltaNet (**Ornith**) | ✅ | ✅ | ✅ MTP + DFlash | ✅ |
| K2-Horizon | ✅ | — | — | ✅ |
| Muse Glimmer | ✅ | ✅ (09-22) | ✅ MTP (single card) | ✅ |
| gemma-4 (incl. head_dim 512) | ✅ | ✅ (default on) | — | ✅ |
| Qwen3.8-Flash-Next (Qwen4-Exp) | ✅ | ✅ (09-22) | ⏳ refused with a drafter | ⏳ refused under TP/PP |
| Agnes-3.0-Flash (parallel-FFN) | ✅ | — | — | ✅ |
| NVFP4 checkpoints (any arch) | ✅ reads → any internal format | — | — | — |

---

## 4. Serving capability (as of `main`)

| Feature | Works | Notes |
|---|---|---|
| Single card, `bin/grimoire` CLI | ✅ | Original path, untouched throughout |
| HTTP server, OpenAI-compatible, SSE streaming | ✅ | `grimoire-server`, `serve.sh` |
| Multi-agent concurrency (batched decode) | ✅ | `GRIMOIRE_SEQ_SLOTS` / `GRIMOIRE_MAX_BATCH`; off by default |
| Conversation resume (prefix cache) | ✅ | `GRIMOIRE_PREFIX_CACHE=1`; now composes with batching |
| Speculative decoding (MTP or DFlash) | ✅ | One card; now composes with batching |
| Two-card pipeline-parallel serving | ✅ | `serve_pp2.sh`; batches given slots since 09-22 |
| Two-card tensor-parallel serving | ✅ | `serve_tp2.sh` (new 09-23); prefix cache works here, unlike PP |
| Batching + speculation together, distributed (PP/TP) | ⏳ | Explicitly refused — needs a real cross-rank scheduling design |
| More than 2 GPUs / mixed GPU models | 🖥️ | Code path exists, never run |

---

## 5. External audits run against this codebase

| # | Date | Scope | Findings | Outcome |
|---|---|---|---|---|
| 1 | 09-12 | First compiler run | Didn't compile; 4 K2 engine bugs incl. a device heap overrun | All fixed |
| 2 | 09-13 | TP/PP scratch buffer | Overrun in tensor-parallel scratch | Fixed, gated |
| 3 | 09-13 | Branch-wide | 6 findings | All fixed |
| 4 | 09-13 | Branch-wide, second pass | 7 findings | All fixed |
| 5 | 09-14 | gemma-4 first pass | Norm convention, value path | Fixed |
| 6 | 09-14/15 | gemma-4 second pass | RoPE factor, graph capture, refusals (5 findings) | All fixed |
| 7 | 09-21 | Whole concurrency/serving layer | **9 defects, 5 P1** (NVFP4 formula inverted, KV-slot collision, PP cancellation desync, snapshot truncation, capacity-check scoping, shell exit-status bug, gate device visibility) | All fixed, independently re-verified |
| 8 | 09-22/23 | Composable-serving commits (this session) | Unsafe partial allocation, TP resume desync risk, O(n) TP allocation, shell exit-status bug (again, different script), untested MTP acceptance path, stale docs | All fixed, negative-controlled |

**Recurring lesson across all 8 audits** (now `CLAUDE.md` rules 15–22): a green gate is a claim about the exact path it drives, not about the feature. The dangerous bugs were never in a feature tested alone — they were two features, each individually gated, breaking only when combined or interrupted mid-flight.

---

## 6. What's left (open items)

1. **Run it on the actual hardware.** Nothing above has touched a B70. This is the next and only mandatory step.
2. Real throughput numbers: tok/s at 1/2/4/8 concurrent requests, batching vs. speculation vs. both, PP vs. TP on OCuLink.
3. Batching + speculation together **under PP/TP** (currently one-card only).
4. Qwen4-Exp under TP/PP, and with a drafter.
5. Speculation depth policy under high concurrency (currently splits draft depth evenly across active rows, which shrinks fast).
6. `GRIMOIRE_DECODE_GRAPH=1` (the single-request decode graph, wired 09-04, still never measured against the baseline it was built to fix).
7. VRAM sizing for `GRIMOIRE_SEQ_SLOTS` against a real checkpoint at real `--ctx`.

---

*Source: `git log` on `origin/main` (256 commits) and the 46 dated `.md` handoffs at the repository root. Full detail for any phase above is in the correspondingly-dated handoff file — see `CLAUDE.md` → "Where to look for current status" for the reading order, and `AUDIT-2026-09-22-TOWER-READINESS.md` for the most recent full audit.*
