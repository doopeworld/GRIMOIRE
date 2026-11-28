#pragma once

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace b70 {

namespace kv_vmm = sycl::ext::oneapi::experimental;

// Stable [slot][head][position][dim] addresses with backing allocated on demand.
// The engine owns sequence lifetimes and must drain the queue before recycling a slot.
class KvVirtualPool {
    struct Reservation {
        sycl::context context;
        size_t bytes;
        uintptr_t base;

        Reservation(const sycl::context& ctx, size_t size)
            : context(ctx), bytes(size), base(kv_vmm::reserve_virtual_mem(bytes, context)) {}
        ~Reservation() {
            try { kv_vmm::free_virtual_mem(base, bytes, context); } catch (...) {}
        }
    };
    struct Mapping {
        sycl::context context;
        size_t bytes;
        kv_vmm::physical_mem backing;
        void* address;

        Mapping(const sycl::device& dev, const sycl::context& ctx, size_t size, uintptr_t base)
            : context(ctx), bytes(size), backing(dev, context, bytes),
              address(backing.map(base, bytes, kv_vmm::address_access_mode::read_write)) {}
        ~Mapping() {
            try { unmap(); } catch (...) {}
        }
        void unmap() {
            if (!address) return;
            kv_vmm::unmap(address, bytes, context);
            address = nullptr;
        }
    };
    struct Buffer {
        size_t heads, dim, stride, divisor;
        std::unique_ptr<Reservation> address;
        size_t rows(size_t tokens) const {
            return divisor == 1 || !tokens ? tokens : tokens / divisor + 1;
        }
    };
    struct Slot {
        size_t tokens = 0;
        size_t bytes = 0;
        std::vector<std::unique_ptr<Mapping>> mapped;
    };
    struct Range {
        uintptr_t base;
        size_t bytes;
    };

    sycl::queue& queue_;
    const sycl::context context_;
    const sycl::device device_;
    const size_t granularity_, capacity_, limit_;
    size_t used_ = 0;
    // Slots destruct first so their mappings are gone before the address ranges.
    std::vector<Buffer> buffers_;
    std::vector<Slot> slots_;

    size_t round_up(size_t value) const {
        return (value + granularity_ - 1) / granularity_ * granularity_;
    }

    std::vector<Range> growth(size_t slot, size_t tokens) const {
        const size_t old = slots_[slot].tokens;
        std::vector<Range> ranges;
        for (const auto& buffer : buffers_) {
            const size_t first_range = ranges.size();
            const size_t head_bytes = buffer.rows(capacity_) * buffer.dim;
            const size_t old_rows = buffer.rows(old);
            const size_t rows = buffer.rows(tokens);
            if (rows == old_rows) continue;
            const size_t base = buffer.address->base + slot * buffer.stride;
            for (size_t head = 0; head < buffer.heads; ++head) {
                const size_t start = head * head_bytes;
                size_t begin = start / granularity_ * granularity_;
                if (old_rows)
                    begin = round_up(start + old_rows * buffer.dim);
                size_t end = round_up(start + rows * buffer.dim);
                // The old prefix of the next head already covers its first page.
                if (old_rows && head + 1 < buffer.heads)
                    end = std::min(end, (start + head_bytes) / granularity_ * granularity_);
                if (end <= begin) continue;
                const uintptr_t first = base + begin;
                if (ranges.size() > first_range && first <= ranges.back().base + ranges.back().bytes) {
                    auto& last = ranges.back();
                    last.bytes = std::max(last.base + last.bytes, base + end) - last.base;
                } else {
                    ranges.push_back({first, end - begin});
                }
            }
        }
        return ranges;
    }

public:
    KvVirtualPool(sycl::queue& queue, size_t capacity, size_t slots, size_t limit)
        : queue_(queue), context_(queue.get_context()), device_(queue.get_device()),
          granularity_(std::max(kv_vmm::get_mem_granularity(device_, context_),
                                kv_vmm::get_mem_granularity(context_))),
          capacity_(capacity), limit_(limit), slots_(slots) {}

    size_t stride(size_t bytes) const { return round_up(bytes); }
    size_t limit() const { return limit_; }
    size_t used() const { return used_; }
    size_t mapped_bytes(size_t slot) const { return slots_[slot].bytes; }

    uint8_t* reserve(size_t heads, size_t dim, size_t divisor = 1) {
        Buffer buffer{heads, dim, 0, divisor, nullptr};
        buffer.stride = stride(heads * dim * buffer.rows(capacity_));
        buffer.address = std::make_unique<Reservation>(context_, buffer.stride * slots_.size());
        auto* base = reinterpret_cast<uint8_t*>(buffer.address->base);
        buffers_.push_back(std::move(buffer));
        return base;
    }

    bool grow(size_t slot, size_t tokens) {
        auto& state = slots_[slot];
        if (!state.tokens && state.bytes) throw std::runtime_error("shared KV slot recycle incomplete");
        if (tokens <= state.tokens) return true;
        auto ranges = growth(slot, std::min(tokens, capacity_));
        size_t bytes = 0;
        for (const auto& range : ranges) bytes += range.bytes;
        if (bytes > limit_ - used_) return false;
        // Xe requires a 2 MiB-aligned virtual address for a 2 MiB allocation.
        // A head's next unmapped page may start at any ordinary page boundary.
        constexpr size_t large_page = size_t(2) << 20;
        const size_t max_extent = std::max(granularity_, (large_page - 1) / granularity_ * granularity_);
        size_t count = 0;
        for (const auto& range : ranges) count += (range.bytes + max_extent - 1) / max_extent;
        state.mapped.reserve(state.mapped.size() + count);
        std::vector<std::unique_ptr<Mapping>> mapped;
        mapped.reserve(count);
        if (!ranges.empty()) queue_.wait_and_throw();
        for (const auto& range : ranges)
            for (size_t offset = 0; offset < range.bytes; offset += max_extent)
                mapped.push_back(std::make_unique<Mapping>(device_, context_,
                    std::min(max_extent, range.bytes - offset), range.base + offset));
        try {
            for (const auto& map : mapped) queue_.memset(map->address, 0, map->bytes);
            if (!mapped.empty()) queue_.wait_and_throw();
        } catch (...) {
            try { queue_.wait(); } catch (...) {}
            throw;
        }
        for (auto& map : mapped) state.mapped.push_back(std::move(map));
        state.tokens = tokens;
        state.bytes += bytes;
        used_ += bytes;
        return true;
    }

    void recycle(size_t slot) {
        auto& state = slots_[slot];
        if (!state.bytes) return;
        queue_.wait_and_throw();
        state.tokens = 0;
        while (!state.mapped.empty()) {
            auto& mapping = state.mapped.back();
            mapping->unmap();
            state.bytes -= mapping->bytes;
            used_ -= mapping->bytes;
            state.mapped.pop_back();
        }
    }

    void zero_buffer(size_t slot, uint8_t* pointer, size_t bytes) {
        const auto start = reinterpret_cast<uintptr_t>(pointer);
        for (const auto& mapping : slots_[slot].mapped) {
            const auto address = reinterpret_cast<uintptr_t>(mapping->address);
            if (address >= start && address - start < bytes)
                queue_.memset(mapping->address, 0, mapping->bytes);
        }
    }
};

} // namespace b70
