// =====================================================================
//  safetensors.cpp
// =====================================================================
#include "b70/safetensors.hpp"
#include "b70/formats.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace b70 {

const char* st_dtype_name(STDtype d) {
    switch (d) {
        case STDtype::F64:  return "F64";
        case STDtype::F32:  return "F32";
        case STDtype::F16:  return "F16";
        case STDtype::BF16: return "BF16";
        case STDtype::F8_E4M3: return "F8_E4M3";
        case STDtype::F8_E5M2: return "F8_E5M2";
        case STDtype::I64:  return "I64";
        case STDtype::I32:  return "I32";
        case STDtype::I16:  return "I16";
        case STDtype::I8:   return "I8";
        case STDtype::U8:   return "U8";
        case STDtype::BOOL: return "BOOL";
        default:            return "?";
    }
}

int st_dtype_size(STDtype d) {
    switch (d) {
        case STDtype::F64: case STDtype::I64: return 8;
        case STDtype::F32: case STDtype::I32: return 4;
        case STDtype::F16: case STDtype::BF16: case STDtype::I16: return 2;
        case STDtype::F8_E4M3: case STDtype::F8_E5M2:
        case STDtype::I8: case STDtype::U8: case STDtype::BOOL: return 1;
        default: return 0;
    }
}

static STDtype parse_dtype(const std::string& s) {
    if (s == "F64")  return STDtype::F64;
    if (s == "F32")  return STDtype::F32;
    if (s == "F16")  return STDtype::F16;
    if (s == "BF16") return STDtype::BF16;
    if (s == "F8_E4M3" || s == "F8_E4M3FN") return STDtype::F8_E4M3;
    if (s == "F8_E5M2") return STDtype::F8_E5M2;
    if (s == "I64")  return STDtype::I64;
    if (s == "I32")  return STDtype::I32;
    if (s == "I16")  return STDtype::I16;
    if (s == "I8")   return STDtype::I8;
    if (s == "U8")   return STDtype::U8;
    if (s == "BOOL") return STDtype::BOOL;
    return STDtype::UNKNOWN;
}

// f16_to_f32 lives in b70/formats.hpp -- one definition, shared with the
// GPTQ decoder, which also reads F16 scales.

void st_to_f32(const void* src, STDtype dtype, int64_t n, float* dst) {
    switch (dtype) {
        case STDtype::F32:
            std::memcpy(dst, src, size_t(n) * 4);
            break;
        case STDtype::F64: {
            const double* p = static_cast<const double*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        case STDtype::BF16: {
            const uint16_t* p = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(bf16_t{p[i]});
            break;
        }
        case STDtype::F16: {
            const uint16_t* p = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(p[i]);
            break;
        }
        case STDtype::F8_E4M3: {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = e4m3_to_f32(p[i]);
            break;
        }
        case STDtype::F8_E5M2: {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = e5m2_to_f32(p[i]);
            break;
        }
        case STDtype::I8: {
            const int8_t* p = static_cast<const int8_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        case STDtype::U8: case STDtype::BOOL: {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        case STDtype::I16: {
            const int16_t* p = static_cast<const int16_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        case STDtype::I32: {
            const int32_t* p = static_cast<const int32_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        case STDtype::I64: {
            const int64_t* p = static_cast<const int64_t*>(src);
            for (int64_t i = 0; i < n; ++i) dst[i] = float(p[i]);
            break;
        }
        default:
            for (int64_t i = 0; i < n; ++i) dst[i] = 0.0f;
    }
}

// ---------------------------------------------------------------------
// Minimal JSON scanner. Only what a safetensors header contains:
// objects, arrays, strings, numbers, true/false/null.
// ---------------------------------------------------------------------
namespace {

struct Json {
    const char* p;
    const char* end;
    bool ok = true;

    void ws() { while (p < end && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) ++p; }
    bool eat(char c) { ws(); if (p < end && *p == c) { ++p; return true; } return false; }
    char peek() { ws(); return p < end ? *p : '\0'; }

    std::string str() {
        std::string out;
        ws();
        if (p >= end || *p != '"') { ok = false; return out; }
        ++p;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                ++p;
                switch (*p) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        // keep it simple: emit '?' for non-ASCII escapes,
                        // tensor names are ASCII in practice
                        if (p + 4 < end) p += 4;
                        out += '?';
                        break;
                    }
                    default: out += *p;
                }
                ++p;
            } else {
                out += *p++;
            }
        }
        if (p >= end) { ok = false; return out; }
        ++p;
        return out;
    }

    double num() {
        ws();
        char* e = nullptr;
        double v = std::strtod(p, &e);
        if (e == p) { ok = false; return 0; }
        p = e;
        return v;
    }

    // Skip any value, used for config keys we do not care about.
    void skip() {
        ws();
        if (p >= end) { ok = false; return; }
        if (*p == '"') { str(); return; }
        if (*p == '{' || *p == '[') {
            const char open = *p, close = (open == '{') ? '}' : ']';
            int depth = 0;
            while (p < end) {
                if (*p == '"') { str(); continue; }
                if (*p == open) ++depth;
                else if (*p == close) { --depth; if (depth == 0) { ++p; return; } }
                ++p;
            }
            ok = false;
            return;
        }
        while (p < end && *p != ',' && *p != '}' && *p != ']') ++p;
    }
};

} // namespace

// ---------------------------------------------------------------------
SafeTensors::~SafeTensors() { close(); }

void SafeTensors::unmap() {
    if (base_) { ::munmap(base_, size_); base_ = nullptr; }
}

void SafeTensors::close() {
    if (base_) { ::munmap(base_, size_); base_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    tensors_.clear();
    meta_.clear();
}

bool SafeTensors::open(const std::string& path, std::string& err) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "cannot open " + path; return false; }

    struct stat st{};
    if (::fstat(fd_, &st) != 0) { err = "fstat failed on " + path; close(); return false; }
    size_ = uint64_t(st.st_size);
    if (size_ < 8) { err = path + " is too small to be safetensors"; close(); return false; }

    base_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (base_ == MAP_FAILED) { base_ = nullptr; err = "mmap failed on " + path; close(); return false; }

    // Weights are read once, sequentially, and never again. Telling the
    // kernel that stops the page cache filling with 30 GB of checkpoint
    // the moment conversion finishes.
    ::madvise(base_, size_, MADV_SEQUENTIAL);

    uint64_t hlen = 0;
    std::memcpy(&hlen, base_, 8);
    if (hlen == 0 || hlen + 8 > size_) {
        err = path + ": bogus header length"; close(); return false;
    }
    data_off_ = 8 + hlen;

    Json j{ static_cast<const char*>(base_) + 8,
            static_cast<const char*>(base_) + 8 + hlen };

    if (!j.eat('{')) { err = path + ": header is not a JSON object"; close(); return false; }
    if (j.peek() != '}') {
        for (;;) {
            const std::string key = j.str();
            if (!j.ok) break;
            if (!j.eat(':')) { j.ok = false; break; }

            if (key == "__metadata__") {
                if (j.eat('{')) {
                    if (j.peek() != '}') {
                        for (;;) {
                            const std::string mk = j.str();
                            if (!j.ok || !j.eat(':')) { j.ok = false; break; }
                            if (j.peek() == '"') meta_[mk] = j.str();
                            else j.skip();
                            if (!j.eat(',')) break;
                        }
                    }
                    j.eat('}');
                }
            } else {
                STTensor t;
                t.name = key;
                if (!j.eat('{')) { j.ok = false; break; }
                for (;;) {
                    const std::string f = j.str();
                    if (!j.ok || !j.eat(':')) { j.ok = false; break; }
                    if (f == "dtype") {
                        t.dtype = parse_dtype(j.str());
                    } else if (f == "shape") {
                        if (!j.eat('[')) { j.ok = false; break; }
                        if (j.peek() != ']')
                            for (;;) { t.shape.push_back(int64_t(j.num()));
                                       if (!j.eat(',')) break; }
                        j.eat(']');
                    } else if (f == "data_offsets") {
                        if (!j.eat('[')) { j.ok = false; break; }
                        t.begin = uint64_t(j.num());
                        j.eat(',');
                        t.end = uint64_t(j.num());
                        j.eat(']');
                    } else {
                        j.skip();
                    }
                    if (!j.eat(',')) break;
                }
                j.eat('}');

                // Validate before trusting the offsets: a truncated shard
                // would otherwise read past the mapping and segfault.
                const int esz = st_dtype_size(t.dtype);
                if (t.dtype == STDtype::UNKNOWN) {
                    err = path + ": tensor '" + t.name + "' has an unsupported dtype";
                    close(); return false;
                }
                if (t.end < t.begin || data_off_ + t.end > size_) {
                    err = path + ": tensor '" + t.name + "' offsets run past end of file";
                    close(); return false;
                }
                if (uint64_t(t.numel()) * uint64_t(esz) != t.end - t.begin) {
                    err = path + ": tensor '" + t.name + "' shape does not match its byte range";
                    close(); return false;
                }
                tensors_[t.name] = t;
            }
            if (!j.eat(',')) break;
        }
    }
    j.eat('}');

    if (!j.ok) { err = path + ": malformed JSON header"; close(); return false; }
    return true;
}

const STTensor* SafeTensors::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

bool SafeTensors::read_raw(const STTensor& t, void* dst, std::string& err) const {
    if (fd_ < 0) { err = "no file open"; return false; }
    size_t   n   = size_t(t.end - t.begin);
    off_t    off = off_t(data_off_ + t.begin);
    uint8_t* p   = static_cast<uint8_t*>(dst);
    while (n > 0) {
        const ssize_t got = ::pread(fd_, p, n, off);
        if (got <= 0) { err = "pread failed on tensor " + t.name; return false; }
        p   += got;
        off += got;
        n   -= size_t(got);
    }
    return true;
}

bool SafeTensors::read_f32(const STTensor& t, float* dst, std::string& err) const {
    if (base_) {
        st_to_f32(data(t), t.dtype, t.numel(), dst);
        return true;
    }
    // Grimoire deliberately releases multi-gigabyte mappings before
    // Level Zero reserves its device VA.  The fd remains open, so retain
    // dtype conversion by reading the tensor bytes through pread.
    std::vector<uint8_t> raw(size_t(t.end - t.begin));
    if (!read_raw(t, raw.data(), err)) return false;
    st_to_f32(raw.data(), t.dtype, t.numel(), dst);
    return true;
}

// ---------------------------------------------------------------------
bool HFModel::discover(const std::string& d, std::string& err) {
    dir = d;
    shards.clear();
    config.clear();

    // ---- config.json -------------------------------------------------
    {
        const std::string cp = dir + "/config.json";
        FILE* f = std::fopen(cp.c_str(), "rb");
        if (!f) { err = "no config.json in " + dir; return false; }
        std::string buf;
        char tmp[8192];
        size_t n;
        while ((n = std::fread(tmp, 1, sizeof tmp, f)) > 0) buf.append(tmp, n);
        std::fclose(f);

        Json j{ buf.data(), buf.data() + buf.size() };
        if (j.eat('{') && j.peek() != '}') {
            for (;;) {
                const std::string k = j.str();
                if (!j.ok || !j.eat(':')) break;
                const char c = j.peek();
                if (c == '"') {
                    config[k] = j.str();
                } else if ((c >= '0' && c <= '9') || c == '-') {
                    const double v = j.num();
                    char b[64];
                    std::snprintf(b, sizeof b, "%.17g", v);
                    config[k] = b;
                } else if (c == 't' || c == 'f') {
                    config[k] = (c == 't') ? "1" : "0";
                    j.skip();
                } else {
                    j.skip();
                }
                if (!j.eat(',')) break;
            }
        }
    }

    // ---- shards ------------------------------------------------------
    // Prefer the index if present; otherwise take every .safetensors in
    // the directory, sorted, so shard order is deterministic.
    DIR* dp = ::opendir(dir.c_str());
    if (!dp) { err = "cannot list " + dir; return false; }
    std::vector<std::string> found;
    while (dirent* de = ::readdir(dp)) {
        const std::string n = de->d_name;
        if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0)
            found.push_back(dir + "/" + n);
    }
    ::closedir(dp);
    if (found.empty()) { err = "no .safetensors files in " + dir; return false; }
    std::sort(found.begin(), found.end());
    shards = found;
    return true;
}

int HFModel::cfg_int(const std::string& key, int fallback) const {
    auto it = config.find(key);
    if (it == config.end()) return fallback;
    return std::atoi(it->second.c_str());
}

std::string HFModel::cfg_str(const std::string& key, const std::string& fallback) const {
    auto it = config.find(key);
    return it == config.end() ? fallback : it->second;
}

} // namespace b70
