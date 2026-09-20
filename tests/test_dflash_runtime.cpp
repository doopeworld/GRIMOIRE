#include "b70/dflash_runtime.hpp"
#include <cassert>
#include <cstdio>

int main() {
    using namespace b70;
    using F = DFlashRuntime::Format;
    const auto a = resolve_dflash_runtime("4", "bf16", false);
    const auto b = resolve_dflash_runtime("8", "fp8", false);
    assert(a.error.empty() && b.error.empty());
    assert(a.rows == 4 && a.format == F::BF16);
    assert(b.rows == 8 && b.format == F::FP8);
    assert(resolve_dflash_runtime(nullptr, nullptr, false).rows == 16);
    assert(resolve_dflash_runtime(nullptr, nullptr, false, 8).rows == 8);
    assert(resolve_dflash_runtime(nullptr, nullptr, true).format == F::BF16);
    assert(resolve_dflash_runtime(nullptr, nullptr, false).format == F::Inherit);
    for (const char* v : {"", "0", "1", "17", "-1", "eight", "8tail", "99999999999999999"})
        assert(!resolve_dflash_runtime(v, nullptr, false).error.empty());
    for (const char* v : {"2", "8", "16"})
        assert(resolve_dflash_runtime(v, nullptr, false).error.empty());
    assert(!resolve_dflash_runtime(nullptr, "mxfp4", false).error.empty());
    assert(!resolve_dflash_runtime(nullptr, "fp8", true).error.empty());
    std::puts("DFlash runtime width/precision isolation: PASS");
}
