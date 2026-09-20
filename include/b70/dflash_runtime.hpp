#pragma once
#include <cstring>
#include <string>

namespace b70 {
// Resolve per engine, never in a process-wide static: several engines may
// use different draft widths/precisions in one resident process.
struct DFlashRuntime {
    enum class Format { Inherit, BF16, FP8 };
    int rows = 16;
    Format format = Format::Inherit;
    std::string error;
};
inline DFlashRuntime resolve_dflash_runtime(const char* rows, const char* format,
                                            bool keep_bf16, int default_rows = 16) {
    DFlashRuntime r;
    r.rows = default_rows;
    if (rows) {
        int value = 0;
        for (const char* p = rows; *p; ++p) {
            if (*p < '0' || *p > '9' || value > 16) {
                r.error = "GRIMOIRE_DFLASH_M must be an integer from 2 to 16";
                return r;
            }
            value = value * 10 + (*p - '0');
        }
        r.rows = value;
    }
    if (r.rows < 2 || r.rows > 16) {
        r.error = "GRIMOIRE_DFLASH_M must be an integer from 2 to 16";
        return r;
    }
    if (!format) r.format = keep_bf16 ? DFlashRuntime::Format::BF16
                                    : DFlashRuntime::Format::Inherit;
    else if (!std::strcmp(format, "bf16")) r.format = DFlashRuntime::Format::BF16;
    else if (!std::strcmp(format, "fp8")) {
        if (keep_bf16) r.error = "GRIMOIRE_DFLASH_DRAFT_FORMAT=fp8 conflicts with GRIMOIRE_DFLASH_DRAFT_BF16";
        else r.format = DFlashRuntime::Format::FP8;
    } else r.error = "GRIMOIRE_DFLASH_DRAFT_FORMAT must be bf16 or fp8";
    return r;
}
} // namespace b70
