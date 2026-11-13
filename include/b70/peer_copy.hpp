// SPDX-License-Identifier: GPL-3.0-or-later
// Opt-in Linux Intel Level Zero IPC transport. GPU payload stays on GPUs;
// Unix sockets carry handles, readiness and consumption acknowledgements only.
// No device-side polling, cross-device atomics, or imported host allocations.
#pragma once
#include "b70/ipc_socket.hpp"
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <level_zero/ze_api.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <vector>

namespace b70 {
class PeerCopy {
    struct Api {
        void* library = nullptr;
        decltype(&zeDeviceGetProperties) properties = nullptr;
        decltype(&zeDeviceGetCommandQueueGroupProperties) groups = nullptr;
        decltype(&zeMemAllocDevice) alloc = nullptr;
        decltype(&zeMemFree) free = nullptr;
        decltype(&zeContextMakeMemoryResident) resident = nullptr;
        decltype(&zeMemGetIpcHandle) get = nullptr;
        decltype(&zeMemPutIpcHandle) put = nullptr;
        decltype(&zeMemOpenIpcHandle) open = nullptr;
        decltype(&zeMemCloseIpcHandle) close = nullptr;
        decltype(&zeCommandListCreateImmediate) create = nullptr;
        decltype(&zeCommandListAppendMemoryCopy) copy = nullptr;
        decltype(&zeCommandListHostSynchronize) sync = nullptr;
        decltype(&zeCommandListDestroy) destroy = nullptr;
        // Leave the default host path free of new loader activity.
        void load() {
            if(library) return;
            library = ::dlopen("libze_loader.so.1", RTLD_NOW | RTLD_LOCAL);
            if (!library) return;
#define B70_PEER_SYMBOL(member, symbol) member = reinterpret_cast<decltype(member)>(::dlsym(library, #symbol))
            B70_PEER_SYMBOL(properties, zeDeviceGetProperties);
            B70_PEER_SYMBOL(groups, zeDeviceGetCommandQueueGroupProperties);
            B70_PEER_SYMBOL(alloc, zeMemAllocDevice); B70_PEER_SYMBOL(free, zeMemFree);
            B70_PEER_SYMBOL(resident, zeContextMakeMemoryResident);
            B70_PEER_SYMBOL(get, zeMemGetIpcHandle); B70_PEER_SYMBOL(put, zeMemPutIpcHandle);
            B70_PEER_SYMBOL(open, zeMemOpenIpcHandle); B70_PEER_SYMBOL(close, zeMemCloseIpcHandle);
            B70_PEER_SYMBOL(create, zeCommandListCreateImmediate);
            B70_PEER_SYMBOL(copy, zeCommandListAppendMemoryCopy);
            B70_PEER_SYMBOL(sync, zeCommandListHostSynchronize);
            B70_PEER_SYMBOL(destroy, zeCommandListDestroy);
#undef B70_PEER_SYMBOL
        }
        bool ready() const { return properties && groups && alloc && free && resident && get && put && open && close && create && copy && sync && destroy; }
        ~Api() { if (library) ::dlclose(library); }
    } api;
    ze_context_handle_t context = nullptr;
    ze_device_handle_t device = nullptr;
    ze_command_list_handle_t copies = nullptr;
    ze_ipc_mem_handle_t exported{};
    bool exported_live = false;
    void* owned = nullptr;
    void* imported = nullptr;
    size_t capacity = 0;
    int socket = -1;
    bool first = false, enabled = false;
    uint64_t sequence = 0;
    struct Hello { uint64_t magic = 0x4752494d50325031ULL; uint64_t bytes = 0; uint32_t mode = 0, available = 0; ze_device_uuid_t uuid{}; };
    struct Packet { uint64_t seq, bytes; uint32_t kind, format; };
    bool native_copy(void* dst, const void* src, size_t bytes) {
        return api.copy(copies, dst, src, bytes, nullptr, 0, nullptr) == ZE_RESULT_SUCCESS &&
               api.sync(copies, 120000000000ULL) == ZE_RESULT_SUCCESS;
    }
    bool exchange_packet(const Packet& own) {
        Packet peer{};
        return ipc::exchange(socket, first, own, peer) && peer.seq == own.seq && peer.bytes == own.bytes &&
               peer.kind == own.kind && peer.format == own.format;
    }
    struct Operation {
        int fd; bool done=false;
        ~Operation() { if(!done) ::shutdown(fd,SHUT_RDWR); }
    };
public:
    PeerCopy() = default;
    PeerCopy(const PeerCopy&) = delete;
    PeerCopy& operator=(const PeerCopy&) = delete;
    ~PeerCopy() { reset(); }
    bool active() const { return enabled; }
    // unset / 0: retain existing host path; auto: use IPC if both imports
    // succeed; 1 / required: require it. A new backend stays opt-in until
    // tested on a physically P2P-capable host. No P2P traffic is needed to
    // decide that a driver's import is unsupported.
    static int requested_mode() {
        const char* e = std::getenv("GRIMOIRE_P2P");
        if (!e || !*e || std::string(e) == "0") return 0;
        if (std::string(e) == "auto") return 1;
        if (std::string(e) == "1" || std::string(e) == "required") return 2;
        return -1;
    }
    // Returns false only for a protocol error or a required but unavailable
    // path. Both endpoints exchange every decision before choosing fallback.
    // Each process retains its per-worker ZE_AFFINITY_MASK. The remote device
    // may be invisible to enumeration; a successful peer IPC import is the
    // capability check enforced by Level Zero / the kernel in that case.
    bool setup(sycl::queue& q, int fd, bool initiator, std::string& reason, bool allow_same_device = false) {
        reset(); socket = fd; first = initiator;
        Hello own{}, peer{};
        const int mode = requested_mode(); own.mode = mode < 0 ? 3u : uint32_t(mode);
        own.bytes = size_t(4) << 20;
        if (const char* e = std::getenv("GRIMOIRE_P2P_CHUNK_MB"))
            own.bytes = size_t(std::clamp(std::atoi(e), 1, 64)) << 20;
        try {
            if(mode>0) api.load();
            if (mode > 0 && api.ready() && q.get_backend() == sycl::backend::ext_oneapi_level_zero) {
                context = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
                device = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
                ze_device_properties_t p{}; p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
                if (api.properties(device, &p) == ZE_RESULT_SUCCESS && p.vendorId == 0x8086) {
                    own.uuid = p.uuid; own.available = 1;
                }
            }
        } catch (const std::exception&) { own.available = 0; }
        if (!ipc::exchange(fd, first, own, peer) || peer.magic != own.magic) { reason = "P2P hello failed"; return false; }
        if (peer.bytes < (size_t(1) << 20) || peer.bytes > (size_t(64) << 20)) { reason = "P2P invalid chunk size"; return false; }
        const bool required = own.mode == 2 || peer.mode == 2;
        if (own.mode == 3 || peer.mode > 2) { reason = "GRIMOIRE_P2P must be 0, auto, or 1 (required)"; return false; }
        auto unavailable = [&](const char* why) { reason = why; reset(); return !required; };
        if (!own.mode || !peer.mode) return unavailable("disabled on one or both ranks");
        if (!own.available || !peer.available) return unavailable("Level Zero IPC API unavailable");
        if (!allow_same_device && std::memcmp(&own.uuid, &peer.uuid, sizeof own.uuid) == 0)
            return unavailable("ranks select the same physical GPU");
        capacity = size_t(std::min(own.bytes, peer.bytes));
        uint32_t ok = 0, peer_ok = 0;
        ze_device_mem_alloc_desc_t md{}; md.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
        if (api.alloc(context, &md, capacity * 2, 65536, device, &owned) == ZE_RESULT_SUCCESS &&
            api.get(context, owned, &exported) == ZE_RESULT_SUCCESS) {
            exported_live = true;
            ok = api.resident(context, device, owned, capacity * 2) == ZE_RESULT_SUCCESS;
        }
        if (!ipc::exchange(fd, first, ok, peer_ok)) { reason = "P2P export agreement failed"; return false; }
        if (!ok || !peer_ok) return unavailable("device allocation export unavailable");
        // On Intel's Linux Level Zero backend, the first int of the opaque
        // IPC handle is a dma-buf fd. SCM_RIGHTS translates it to the receiver's
        // fd table; the remaining opaque bytes are preserved verbatim.
        ze_ipc_mem_handle_t peer_handle{}; int export_fd = -1, received_fd = -1;
        std::memcpy(&export_fd, &exported, sizeof export_fd);
        auto send_handle = [&] { return export_fd >= 0 && ipc::send_fd(fd, export_fd) && ipc::write_all(fd, &exported, sizeof exported); };
        auto recv_handle = [&] { return ipc::recv_fd(fd, received_fd) && ipc::read_all(fd, &peer_handle, sizeof peer_handle); };
        const bool handles_ok = first ? send_handle() && recv_handle() : recv_handle() && send_handle();
        if (!handles_ok) { if (received_fd >= 0) ::close(received_fd); reason = "P2P handle exchange failed"; return false; }
        std::memcpy(&peer_handle, &received_fd, sizeof received_fd);
        // Flags 0, not ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED: MEASURED 2026-11-13 (two-process Level Zero test, both
        // cards enumerated, P2PDMA kernel) -- the uncached import is never resident (0x70000003), the default one
        // is and 64 KB pushes land correctly (39-47 us).  The copy engine only WRITES through this mapping.
        ok = api.open(context, device, peer_handle, 0, &imported) == ZE_RESULT_SUCCESS;
        ::close(received_fd);
        // The import must be mappable here.  MEASURED 2026-11-13 (Tower, P2PDMA kernel): with one card per
        // process (ZE_AFFINITY_MASK=r) zeMemOpenIpcHandle succeeds but residency fails (0x70000003) and copies
        // into the import never run -- both ranks stalled.  With both cards enumerated it maps and works
        // (docker-entrypoint.sh: GRIMOIRE_MULTI_SEE_ALL=1).  Checked here so that case falls back instead.
        bool not_resident = false;
        if (ok && api.resident(context, device, imported, capacity * 2) != ZE_RESULT_SUCCESS) { ok = false; not_resident = true; }
        if (ok) {
            uint32_t n = 0; ok = api.groups(device, &n, nullptr) == ZE_RESULT_SUCCESS && n;
            std::vector<ze_command_queue_group_properties_t> groups(n);
            for (auto& p : groups) p.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
            if (ok) ok = api.groups(device, &n, groups.data()) == ZE_RESULT_SUCCESS;
            uint32_t ordinal = n;
            for (uint32_t i = 0; ok && i < n; ++i)
                if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) && groups[i].numQueues) { ordinal = i; break; }
            ze_command_queue_desc_t cd{}; cd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
            cd.ordinal = ordinal; cd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            ok = ok && ordinal < n && api.create(context, device, &cd, &copies) == ZE_RESULT_SUCCESS;
        }
        if (!ipc::exchange(fd, first, ok, peer_ok)) { reason = "P2P import agreement failed"; return false; }
        if (!ok || !peer_ok)
            return unavailable(not_resident ? "peer memory import not resident here (each rank must see both cards: "
                                              "GRIMOIRE_MULTI_SEE_ALL=1)"
                                            : "peer IPC import unsupported by driver/topology");
        enabled = true; reason = "direct Level Zero IPC peer copies";
        return true;
    }
    // Two-rank all-reduce. Each rank snapshots before either reads its peer,
    // and both acknowledge completion before either reuses the exported slot.
    // Only owned local memory is accessed by kernels; the copy engine reads
    // the peer's IPC mapping. No peer-visible flags share the data allocation.
    bool allreduce(sycl::queue& q, float* data, size_t elems, int rank, bool bf16, const float* host_add = nullptr) {
        if (!enabled) return false;
        Operation op{socket};
        const size_t width = bf16 ? 2 : 4, limit = capacity / width;
        for (size_t at = 0; at < elems; at += limit) {
            const size_t n = std::min(limit, elems - at); float* src = data + at;
            q.wait_and_throw();
            const float* extra = host_add ? host_add + at : nullptr;
            auto own = static_cast<uint8_t*>(owned);
            auto other = own + capacity;
            if (bf16) {
                auto out = reinterpret_cast<uint16_t*>(own);
                q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    const float v = extra ? src[i] + extra[i] : src[i];
                    uint32_t u = sycl::bit_cast<uint32_t>(v);
                    out[i] = uint16_t((u & 0x7fffffffu) > 0x7f800000u ? (u >> 16) | 0x40u :
                                     (u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
                }).wait_and_throw();
            } else if (extra) {
                auto out = reinterpret_cast<float*>(own);
                q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { out[i] = src[i] + extra[i]; }).wait_and_throw();
            }
            // PUSH (2026-11-13): this rank's copy engine WRITES its partial into the peer's receive half
            // (imported + capacity) instead of reading the peer's send half.  MEASURED on the Tower (Core
            // Ultra 200H root complex, custom P2PDMA kernel): peer writes work (64 KB checked, 2.43 GB/s), the
            // old pull (a P2P read) stalled both ranks in the warm-up with the engines idle.  An fp32 partial
            // with nothing to add goes straight from src.
            const void* payload = (bf16 || extra) ? static_cast<const void*>(own) : static_cast<const void*>(src);
            if (!native_copy(static_cast<uint8_t*>(imported) + capacity, payload, n * width)) return false;
            const Packet ready{++sequence, n * width, 1, uint32_t(bf16)};
            if (!exchange_packet(ready)) return false;   // both partials have landed in the receive halves
            const bool rank0 = rank == 0;
            if (bf16) {
                const auto a = reinterpret_cast<const uint16_t*>(rank0 ? own : other);
                const auto b = reinterpret_cast<const uint16_t*>(rank0 ? other : own);
                q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                    src[i] = sycl::bit_cast<float>(uint32_t(a[i]) << 16) + sycl::bit_cast<float>(uint32_t(b[i]) << 16);
                }).wait_and_throw();
            } else {
                // fp32 a + b == b + a, so reading this rank's partial in place (src, or own when a CPU partial
                // was added) gives both ranks the same bits
                const float* mine = extra ? reinterpret_cast<const float*>(own) : src;
                const auto peer_part = reinterpret_cast<const float*>(other);
                q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { src[i] = mine[i] + peer_part[i]; }).wait_and_throw();
            }
            // both ranks have read their receive half: the next chunk / call may overwrite it
            const Packet consumed{sequence, n * width, 2, uint32_t(bf16)};
            if (!exchange_packet(consumed)) return false;
        }
        op.done=true; return true;
    }
    bool send(sycl::queue& q, const float* src, size_t elems) {
        if (!enabled || elems > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
        Operation op{socket};
        const size_t bytes = elems * sizeof(float);
        q.wait_and_throw();
        for (size_t at = 0; at < bytes; at += capacity) {
            const size_t n = std::min(capacity, bytes - at);
            // push into the receiver's buffer (a P2P write; reads of the peer stalled on the Tower, 11-13)
            if (!native_copy(imported, reinterpret_cast<const uint8_t*>(src) + at, n)) return false;
            Packet msg{++sequence, n, 3, 0}, ack{};
            if (!ipc::write_all(socket, &msg, sizeof msg) || !ipc::read_all(socket, &ack, sizeof ack) ||
                ack.seq != msg.seq || ack.bytes != n || ack.kind != 4 || ack.format != 0) return false;
        }
        op.done=true; return true;
    }
    bool recv(sycl::queue& q, float* dst, size_t elems) {
        if (!enabled || elems > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
        Operation op{socket};
        const size_t bytes = elems * sizeof(float);
        q.wait_and_throw();
        for (size_t at = 0; at < bytes; at += capacity) {
            const size_t n = std::min(capacity, bytes - at); Packet msg{};
            if (!ipc::read_all(socket, &msg, sizeof msg) || msg.seq != ++sequence || msg.bytes != n ||
                msg.kind != 3 || msg.format != 0 || !native_copy(reinterpret_cast<uint8_t*>(dst) + at, owned, n)) return false;
            const Packet ack{sequence, n, 4, 0};
            if (!ipc::write_all(socket, &ack, sizeof ack)) return false;
        }
        op.done=true; return true;
    }
    // Call only after queues have drained and before the owning context dies.
    void reset() {
        enabled = false;
        if (copies) { api.sync(copies, std::numeric_limits<uint64_t>::max()); api.destroy(copies); copies = nullptr; }
        if (imported) { api.close(context, imported); imported = nullptr; }
        if (exported_live) { api.put(context, exported); exported_live = false; }
        if (owned) { api.free(context, owned); owned = nullptr; }
        capacity = 0; sequence = 0;
    }
};
} // namespace b70
