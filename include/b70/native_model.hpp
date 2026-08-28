#ifndef B70_NATIVE_MODEL_HPP
#define B70_NATIVE_MODEL_HPP

#include <cstdint>
#include <string>
#include <unordered_map>

namespace b70 {

constexpr uint64_t kNativeMagic = 0x314c444f4d303742ull; // "B70MODL1"
constexpr uint32_t kNativeVersion = 2;
constexpr uint32_t kNativeAlignment = 4096;

enum class NativeEncoding : uint32_t {
    RAW = 0,
    // Canonical Xe2 kernel input: packed E2M1 rows [N,K/2] and E8M0 scales [N,K/32].
    MXFP4_GRIMOIRE_XE2 = 1,
};

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

class NativeModel {
  public:
    void drop_resident();   // MADV_DONTNEED resident pages, keep mapping
public:
    ~NativeModel();
    bool open(const std::string& path, std::string& err);
    void close();
    const NativeTensorRecord* find(const std::string& name) const;
    const void* payload(const NativeTensorRecord& r) const;
    const void* scales(const NativeTensorRecord& r) const;
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
