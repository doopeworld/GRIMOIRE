#!/bin/bash
# Negative control: the same pool with relaxed atomics on enter/exit/open must trip the sanitizer. No GPU.
set -u
W=/mnt/storage/isos/grimoire-cxpool-1102/tree
sed -e "s/state_.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel, std::memory_order_acquire)/state_.compare_exchange_weak(s, s + 1, std::memory_order_relaxed, std::memory_order_relaxed)/" \
    -e "s/state_.fetch_sub(1, std::memory_order_release)/state_.fetch_sub(1, std::memory_order_relaxed)/" \
    -e "s/state_.store(g << CX_GEN_SHIFT | CX_OPEN, std::memory_order_release)/state_.store(g << CX_GEN_SHIFT | CX_OPEN, std::memory_order_relaxed)/" \
    -e "s/while (state_.load(std::memory_order_acquire) \& CX_INSIDE)/while (state_.load(std::memory_order_relaxed) \& CX_INSIDE)/" \
    -e "s/state_.fetch_and(~CX_OPEN, std::memory_order_acq_rel)/state_.fetch_and(~CX_OPEN, std::memory_order_relaxed)/" \
    -e "s/uint64_t s = state_.load(std::memory_order_acquire);/uint64_t s = state_.load(std::memory_order_relaxed);/" \
    -e "s/s = state_.load(std::memory_order_acquire);/s = state_.load(std::memory_order_relaxed);/" \
    "$W/src/cpu_experts.cpp" > "$W/bin/cpu_experts_broken.cpp"
diff "$W/src/cpu_experts.cpp" "$W/bin/cpu_experts_broken.cpp" | grep -c "^>"
docker run --rm --name grim-cxb-tsanc --cpuset-cpus 0-13 --security-opt seccomp=unconfined -v "$W":/grimoire -w /grimoire grimoire-dev:latest bash -lc "
  source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
  icpx -fsycl -fsanitize=thread -g -O1 -std=c++20 -I include -I src bin/cxpool_tsan.cpp bin/cpu_experts_broken.cpp -lpthread -o bin/cxpool_tsan_broken 2>&1 | tail -5
  TSAN_OPTIONS=halt_on_error=0:report_signal_unsafe=0 timeout 100 setarch -R bin/cxpool_tsan_broken 2>&1 | grep -E \"threads|POOL_STRESS|WARNING: ThreadSanitizer|SUMMARY\" | sed -E \"s/\\(pid=[0-9]+\\)//\" | cut -c1-110 | sort | uniq -c | sort -rn | head -10
  echo control done
" | cut -c1-170
rm -f "$W/bin/cpu_experts_broken.cpp"
