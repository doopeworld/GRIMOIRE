// =====================================================================
//  b70/safetensors.hpp  --  zero-copy safetensors reader
//
//  Layout:  [u64 header_len][header_len bytes of JSON][raw tensor data]
//
//  The file is mmap'd, so the tensor bytes are never copied on the host;
//  the converter reads them straight out of the page cache and writes
//  quantized output. A 14 GB fp16 checkpoint converts without ever
//  holding 14 GB of RSS.
//
//  No JSON library: the header schema is small and fixed, so a ~150 line
//  recursive parser beats adding a dependency to a "bare-metal" project.
// =====================================================================
#ifndef B70_SAFETENSORS_HPP
#define B70_SAFETENSORS_HPP

#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace b70 {

enum class STDtype { F64, F32, F16, BF16, F8_E4M3, F8_E5M2,
                     I64, I32, I16, I8, U8, BOOL, UNKNOWN };

struct STTensor {
    std::string          name;
    STDtype              dtype = STDtype::UNKNOWN;
    std::vector<int64_t> shape;
    uint64_t             begin = 0, end = 0;   // offsets into the data segment

    int64_t numel() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

const char* st_dtype_name(STDtype d);
int         st_dtype_size(STDtype d);

// Convert `n` elements of any supported dtype to fp32.
void st_to_f32(const void* src, STDtype dtype, int64_t n, float* dst);

class SafeTensors {
public:
    ~SafeTensors();
    // Returns false and fills `err` rather than throwing; the converter
    // walks many shard files and one bad shard should name itself.
    bool open(const std::string& path, std::string& err);
    void close();

    const std::map<std::string, STTensor>& tensors() const { return tensors_; }
    const STTensor* find(const std::string& name) const;

    // Raw pointer into the mmap for a tensor. Valid until close().
    const void* data(const STTensor& t) const {
        return static_cast<const uint8_t*>(base_) + data_off_ + t.begin;
    }

    // Read a tensor's raw bytes via pread. Does NOT touch the mmap, so
    // it is immune to whatever page-residency or mapping problem makes
    // dereferencing base_ unreliable in this process.
    bool read_raw(const STTensor& t, void* dst, std::string& err) const;

    // Release the mapping but keep the file descriptor. Once every read
    // goes through read_raw the mapping is dead weight -- and 16 shards
    // hold ~21 GB of virtual mappings, which collides with the address
    // space the Level Zero driver reserves for device allocations.
    void unmap();

    int  shard_fd() const { return fd_; }
    uint64_t data_offset() const { return data_off_; }

    // Read a tensor as fp32 into `dst` (must hold t.numel() floats).
    bool read_f32(const STTensor& t, float* dst, std::string& err) const;

    const std::map<std::string, std::string>& metadata() const { return meta_; }
    uint64_t file_size() const { return size_; }

private:
    void*    base_     = nullptr;
    uint64_t size_     = 0;
    uint64_t data_off_ = 0;
    int      fd_       = -1;
    std::map<std::string, STTensor>    tensors_;
    std::map<std::string, std::string> meta_;
};

// A HuggingFace model directory: config.json plus one or more shards
// listed in model.safetensors.index.json (or a single model.safetensors).
struct HFModel {
    std::string dir;
    std::vector<std::string> shards;
    std::map<std::string, std::string> config;   // flat scalars from config.json

    bool discover(const std::string& dir, std::string& err);
    int  cfg_int(const std::string& key, int fallback) const;
    std::string cfg_str(const std::string& key, const std::string& fallback) const;
};

} // namespace b70
#endif
