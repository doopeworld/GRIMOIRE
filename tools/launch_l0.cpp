// =====================================================================
//  launch_l0.cpp -- what does one dispatch cost, SYCL vs raw Level Zero?
//
//  Decode submits 1267 kernels per token and the projected-cost model in
//  grimoire_load_report charges a FLAT 5 us each for launch.  That number
//  was never measured.  Everything about "use Level Zero for TG" rests on
//  it, so measure it, against the same operation on both paths.
//
//  The operation compared is identical on both sides: a device memory
//  fill of `bytes`.  Same engine, same data, only the dispatch path
//  differs.  Plus an empty SYCL kernel for the pure-submit floor.
//
//  RULES LEARNED THE HARD WAY (first version wedged a GPU for 5 minutes):
//    * stdout UNBUFFERED -- a hang must not swallow the output that says
//      where it hung.
//    * NEVER an infinite Level Zero wait.  Every synchronize gets a finite
//      timeout and reports TIMEOUT instead of spinning a core forever.
// =====================================================================
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::high_resolution_clock;

static constexpr uint64_t kWaitNs = 10ull * 1000 * 1000 * 1000;   // 10 s, never UINT64_MAX

static bool zok(ze_result_t r, const char* what, int line) {
    if (r == ZE_RESULT_SUCCESS) return true;
    std::printf("  !! %s = 0x%x (line %d)\n", what, (unsigned)r, line);
    return false;
}
#define ZCHK(x) zok((x), #x, __LINE__)

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int  N     = (argc > 1) ? std::atoi(argv[1]) : 2000;
    const size_t KB  = (argc > 2) ? size_t(std::atoi(argv[2])) : 256;
    const size_t bytes = KB * 1024;

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("driver : %s\n", q.get_device().get_info<sycl::info::device::driver_version>().c_str());
    std::printf("N=%d  op=%zu KB fill\n\n", N, KB);

    uint8_t* buf = sycl::malloc_device<uint8_t>(bytes, q);
    q.memset(buf, 0, bytes).wait();

    auto per_op = [&](const char* tag, auto&& body) {
        body(50);                                   // warm
        const auto t0 = clk::now();
        body(N);
        const auto t1 = clk::now();
        const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        std::printf("  %-46s %8.3f us\n", tag, us);
        return us;
    };

    // ---- 1. SYCL: empty kernel, in-order, one drain -------------------
    const double sycl_empty = per_op("SYCL empty kernel, in-order, batch+drain",
        [&](int reps) {
            for (int i = 0; i < reps; ++i)
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {}); });
            q.wait();
        });

    // ---- 2. SYCL: the fill --------------------------------------------
    const double sycl_fill = per_op("SYCL memset, in-order, batch+drain",
        [&](int reps) {
            for (int i = 0; i < reps; ++i) q.memset(buf, i & 0xff, bytes);
            q.wait();
        });

    // ---- 3. SYCL: fill, drained EVERY op (round-trip latency) ----------
    per_op("SYCL memset, drain every op",
        [&](int reps) {
            for (int i = 0; i < reps; ++i) q.memset(buf, i & 0xff, bytes).wait();
        });

    // ---- raw Level Zero on the SAME device/context ---------------------
    std::printf("\n  ---- raw Level Zero ----\n");
    if (!ZCHK(zeInit(0))) { sycl::free(buf, q); return 0; }
    auto zectx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto zedev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    std::printf("  interop ctx=%p dev=%p\n", (void*)zectx, (void*)zedev);

    uint32_t ngrp = 0;
    if (!ZCHK(zeDeviceGetCommandQueueGroupProperties(zedev, &ngrp, nullptr))) return 0;
    std::vector<ze_command_queue_group_properties_t> grp(ngrp);
    for (auto& g : grp) g = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr, 0, 0, 0};
    ZCHK(zeDeviceGetCommandQueueGroupProperties(zedev, &ngrp, grp.data()));
    uint32_t ordinal = 0;
    for (uint32_t i = 0; i < ngrp; ++i)
        if (grp[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) { ordinal = i; break; }
    std::printf("  compute ordinal %u of %u groups (%u queues)\n",
                ordinal, ngrp, grp[ordinal].numQueues);

    // One event to close each batch.  Waiting on an event we ourselves
    // signalled is the only wait shape that cannot hang on an empty list.
    ze_event_pool_desc_t epd = {ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
                                ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
    ze_event_pool_handle_t pool = nullptr;
    if (!ZCHK(zeEventPoolCreate(zectx, &epd, 1, &zedev, &pool))) return 0;
    ze_event_desc_t ed = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0,
                          0, ZE_EVENT_SCOPE_FLAG_HOST};
    ze_event_handle_t ev = nullptr;
    if (!ZCHK(zeEventCreate(pool, &ed, &ev))) return 0;

    const uint8_t pat = 0xAB;

    // ---- 4. L0 immediate command list, in-order ------------------------
    {
        ze_command_queue_desc_t cqd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
            ordinal, 0, ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
            ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_list_handle_t imm = nullptr;
        if (ZCHK(zeCommandListCreateImmediate(zectx, zedev, &cqd, &imm))) {
            bool ok = true;
            auto body = [&](int reps) {
                if (!ok) return;
                zeEventHostReset(ev);
                for (int i = 0; i < reps - 1; ++i)
                    if (!ZCHK(zeCommandListAppendMemoryFill(imm, buf, &pat, 1, bytes,
                                                            nullptr, 0, nullptr))) { ok = false; return; }
                if (!ZCHK(zeCommandListAppendMemoryFill(imm, buf, &pat, 1, bytes,
                                                        ev, 0, nullptr))) { ok = false; return; }
                if (!ZCHK(zeEventHostSynchronize(ev, kWaitNs))) { ok = false; return; }
            };
            body(50);
            if (ok) {
                const auto t0 = clk::now();
                body(N);
                const auto t1 = clk::now();
                if (ok) std::printf("  %-46s %8.3f us\n", "L0 immediate list, in-order, batch+event",
                       std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
            }
            zeCommandListDestroy(imm);
        }
    }

    // ---- 5. L0 regular list, recorded ONCE, executed N times -----------
    {
        ze_command_queue_desc_t qd = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr,
            ordinal, 0, 0, ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
            ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
        ze_command_queue_handle_t cq = nullptr;
        ze_command_list_desc_t cld = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr,
                                      ordinal, ZE_COMMAND_LIST_FLAG_IN_ORDER};
        ze_command_list_handle_t cl = nullptr;
        if (ZCHK(zeCommandQueueCreate(zectx, zedev, &qd, &cq)) &&
            ZCHK(zeCommandListCreate(zectx, zedev, &cld, &cl))) {
            const int CHAIN = 64;
            bool ok = true;
            for (int i = 0; i < CHAIN && ok; ++i)
                ok = ZCHK(zeCommandListAppendMemoryFill(cl, buf, &pat, 1, bytes,
                                                        nullptr, 0, nullptr));
            ok = ok && ZCHK(zeCommandListClose(cl));
            auto body = [&](int reps) {
                if (!ok) return;
                const int iters = (reps + CHAIN - 1) / CHAIN;
                for (int i = 0; i < iters && ok; ++i)
                    ok = ZCHK(zeCommandQueueExecuteCommandLists(cq, 1, &cl, nullptr));
                ok = ok && ZCHK(zeCommandQueueSynchronize(cq, kWaitNs));
            };
            body(CHAIN * 2);
            if (ok) {
                const auto t0 = clk::now();
                body(N);
                const auto t1 = clk::now();
                const int done = ((N + CHAIN - 1) / CHAIN) * CHAIN;
                if (ok) std::printf("  %-46s %8.3f us\n", "L0 prebuilt list (64 ops), replayed",
                       std::chrono::duration<double, std::micro>(t1 - t0).count() / done);
            }
            if (cl) zeCommandListDestroy(cl);
            if (cq) zeCommandQueueDestroy(cq);
        }
    }

    std::printf("\n  ---- what this means for decode -------------------\n");
    std::printf("  1267 launches/token at the SYCL empty-kernel floor : %6.2f ms/token\n",
                1267 * sycl_empty / 1000.0);
    std::printf("  SYCL fill minus empty (the fill's own work)        : %6.3f us\n",
                sycl_fill - sycl_empty);

    zeEventDestroy(ev);
    zeEventPoolDestroy(pool);
    sycl::free(buf, q);
    return 0;
}
