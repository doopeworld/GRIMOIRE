// GRIMOIRE -- host-only GGUF v3 reader. SPDX-License-Identifier: GPL-3.0-or-later
#ifndef B70_GGUF_HPP
#define B70_GGUF_HPP

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace b70 {

// GGUF dimensions are innermost first: ne[0] is contiguous K (input
// features), ne[1] is N rows, and ne[2] is the expert index for 3-D weights.
// Shapes are reported as stored; ffn_gate_inp_shexp may be [2048] or [2048,1].
struct GGUFTensor {
    std::string name;
    std::array<uint64_t, 4> ne{{1, 1, 1, 1}};
    uint32_t n_dims = 0, type = 0;
    uint64_t offset = 0; // absolute file offset
    uint64_t nbytes = 0, n_rows = 0, row_bytes = 0;
    uint32_t shard = 0;
};

struct GGUFTypeInfo {
    uint32_t id, block_elems, block_bytes;
    const char* name;
};
const GGUFTypeInfo* gguf_type_info(uint32_t id);
// Select one text model in a directory, or retain an explicitly named file.
// An empty result means the directory contains no text GGUF.
bool gguf_model_path(const std::string& input, std::string& selected, std::string& err);

// Values retain their GGUF type, including the element type of arrays.
// bool arrays use uint8_t values (0 or 1), avoiding vector<bool> proxies.
using GGUFData = std::variant<uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float, bool, std::string, std::vector<uint8_t>,
    std::vector<int8_t>, std::vector<uint16_t>, std::vector<int16_t>,
    std::vector<uint32_t>, std::vector<int32_t>, std::vector<float>,
    std::vector<std::string>, uint64_t, int64_t, double,
    std::vector<uint64_t>, std::vector<int64_t>, std::vector<double>>;
struct GGUFValue {
    uint32_t type = 0, element_type = 0; // element_type only for ARRAY
    GGUFData data{};
    uint64_t array_size() const;
};

class GGUFFile {
public:
    GGUFFile() = default;
    ~GGUFFile();
    GGUFFile(const GGUFFile&) = delete;
    GGUFFile& operator=(const GGUFFile&) = delete;

    // header_only skips only tensor data versus file-size validation. It is
    // intended for truncated header slices, not for reading tensor payloads.
    bool open(const std::string& path, std::string& err, bool header_only = false);
    void close();
    uint32_t version() const { return version_; }
    uint64_t alignment() const { return alignment_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }
    const std::map<std::string, GGUFValue>& metadata() const { return metadata_; }
    const GGUFValue* get(const std::string& key) const;
    bool get_u32(const std::string& key, uint32_t& out, std::string& err) const;
    bool get_u64(const std::string& key, uint64_t& out, std::string& err) const;
    bool get_f32(const std::string& key, float& out, std::string& err) const;
    bool get_string(const std::string& key, std::string& out, std::string& err) const;
    bool get_array(const std::string& key, const GGUFValue*& out, std::string& err) const;
    const std::vector<GGUFTensor>& tensors() const { return tensors_; }
    const GGUFTensor* find(const std::string& name) const;
    bool read_raw(const GGUFTensor& tensor, void* dst, std::string& err) const;
    bool read_raw_row(const GGUFTensor& tensor, uint64_t row, void* dst, std::string& err) const;

private:
    bool open_single(const std::string& path, std::string& err, bool header_only);
    int fd_ = -1;
    uint64_t size_ = 0, alignment_ = 32, data_start_ = 0;
    uint32_t version_ = 0;
    std::map<std::string, GGUFValue> metadata_;
    std::vector<GGUFTensor> tensors_;
    std::map<std::string, size_t> tensor_index_;
    std::vector<std::unique_ptr<GGUFFile>> shards_;
};

} // namespace b70
#endif
