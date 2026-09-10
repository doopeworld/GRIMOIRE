#ifndef B70_NATIVE_MODEL_HPP
#define B70_NATIVE_MODEL_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include "weights.hpp"

namespace b70 {

constexpr uint64_t kNativeMagic = 0x314c444f4d303742ull; // "B70MODL1"
constexpr uint32_t kNativeVersion = 3;
constexpr uint32_t kNativeAlignment = 4096;

enum class NativeEncoding : uint32_t {
    RAW = 0,
    // Canonical Xe2 kernel input: packed E2M1 rows [N,K/2] and E8M0 scales [N,K/32].
    MXFP4_GRIMOIRE_XE2 = 1,
    INT4_AFFINE_G128 = 2,
    FP8_E4M3_CHANNEL = 3,
    FP8_E5M2_CHANNEL = 4,
    INT8_CHANNEL = 5,
    MXFP8_GRIMOIRE_XE2 = 6,
};

// v3 keeps the v2 record ABI. INT4's scales region contains all BF16
// scales first, then all uint8 zero points. Other encodings have no zeros.
struct NativeLayout {
    Fmt fmt = Fmt::BF16;
    int N = 0, K = 0;
    uint64_t row_payload_bytes = 0, row_scale_bytes = 0, row_zero_bytes = 0;
    uint64_t payload_bytes = 0, scale_bytes = 0, zero_bytes = 0;
};
NativeEncoding native_encoding(Fmt fmt);
const char* native_encoding_name(uint32_t encoding);
size_t scale_element_bytes(Fmt fmt);

struct NativeFileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t alignment;
    uint64_t tensor_count;
    uint64_t toc_offset;
    uint64_t file_size;
    uint32_t target;       // 0x031 = intel_gpu_bmg_g31
    uint32_t reserved;
};

struct NativeTensorRecord {
    char name[192];
    uint32_t encoding;
    uint32_t source_dtype;
    uint32_t rank;
    uint32_t flags;
    int64_t shape[4];
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t scales_offset;
    uint64_t scales_bytes;
    int32_t tile_n;
    int32_t tile_k;
    int32_t padded_n;
    int32_t padded_k;
};

static_assert(sizeof(NativeFileHeader) == 48);
static_assert(sizeof(NativeTensorRecord) == 288);

bool native_layout(const NativeTensorRecord& r, NativeLayout& out, std::string& err);
// Copy an already packed view without quantization or precision changes.
PackedWeight copy_packed(const QuantWeight& w);
QuantWeight slice_quant_rows(const QuantWeight& w, int first_row, int rows);

class NativeModel {
  public:
    void drop_resident();   // MADV_DONTNEED resident pages, keep mapping
public:
    NativeModel() = default;
    NativeModel(const NativeModel&) = delete;
    NativeModel& operator=(const NativeModel&) = delete;
    ~NativeModel();
    bool open(const std::string& path, std::string& err);
    void close();
    const NativeTensorRecord* find(const std::string& name) const;
    const void* payload(const NativeTensorRecord& r) const;
    const void* scales(const NativeTensorRecord& r) const;
    bool view(const NativeTensorRecord& r, uint64_t first_row, int rows,
              QuantWeight& out, std::string& err) const;
    const NativeTensorRecord* records() const { return toc_; }
    const NativeFileHeader& header() const { return header_; }
private:
    int fd_=-1;
    void* base_=nullptr;
    uint64_t mapped_size_=0;
    NativeFileHeader header_{};
    const NativeTensorRecord* toc_=nullptr;
    std::unordered_map<std::string,const NativeTensorRecord*> index_;
};

} // namespace b70
#endif
