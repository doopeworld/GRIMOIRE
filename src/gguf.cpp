// GRIMOIRE -- strict host GGUF v3 reader. SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf.hpp"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace b70 {
namespace {
constexpr uint64_t MAX_STRING = 1ull << 26;
constexpr uint64_t MAX_HEADER = 1ull << 30;
constexpr uint64_t MAX_TENSORS = 1ull << 20;
constexpr uint64_t MAX_KV = 1ull << 16;
constexpr uint64_t MAX_ARRAY_ITEMS = 1ull << 22;

uint64_t checked_add(uint64_t a, uint64_t b, const char* what) {
    if (b > UINT64_MAX - a) throw std::runtime_error(std::string(what) + " overflow");
    return a + b;
}
uint64_t checked_mul(uint64_t a, uint64_t b, const char* what) {
    if (b && a > UINT64_MAX / b) throw std::runtime_error(std::string(what) + " overflow");
    return a * b;
}
// Buffer only the header/metadata/tensor-info region, fetching it in chunks.
struct Reader {
    int fd;
    uint64_t file_size, pos = 0;
    std::vector<uint8_t> bytes;
    void need(uint64_t n) {
        if (n > file_size - pos) throw std::runtime_error("truncated GGUF header");
        const uint64_t end = pos + n;
        if (end > MAX_HEADER) throw std::runtime_error("GGUF header exceeds 1 GiB limit");
        while (bytes.size() < end) {
            const uint64_t target = std::min<uint64_t>(file_size,
                std::min<uint64_t>(MAX_HEADER, std::max<uint64_t>(end, bytes.size() + 65536)));
            const size_t old = bytes.size();
            bytes.resize(static_cast<size_t>(target));
            size_t got = 0;
            while (got < target - old) {
                ssize_t r = ::pread(fd, bytes.data() + old + got, target - old - got,
                                    static_cast<off_t>(old + got));
                if (r < 0 && errno == EINTR) continue;
                if (r <= 0) throw std::runtime_error("pread failed in GGUF header");
                got += static_cast<size_t>(r);
            }
        }
    }
    uint64_t u(unsigned n) {
        need(n); uint64_t v = 0;
        for (unsigned i = 0; i < n; ++i) v |= uint64_t(bytes[pos++]) << (8 * i);
        return v;
    }
    std::string str() {
        const uint64_t n = u(8);
        if (n > MAX_STRING) throw std::runtime_error("GGUF string exceeds 64 MiB cap");
        if (n > file_size - pos) throw std::runtime_error("GGUF string length past EOF");
        need(n);
        std::string s(reinterpret_cast<const char*>(bytes.data() + pos), static_cast<size_t>(n));
        pos += n; return s;
    }
    float f32() { uint32_t x = static_cast<uint32_t>(u(4)); float f; std::memcpy(&f, &x, 4); return f; }
    double f64() { uint64_t x = u(8); double f; std::memcpy(&f, &x, 8); return f; }
    bool boolean() {
        uint64_t v = u(1);
        if (v > 1) throw std::runtime_error("invalid GGUF bool value");
        return v != 0;
    }
};

uint64_t element_bytes(uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 8: case 10: case 11: case 12: return 8;
        default: throw std::runtime_error("unknown GGUF metadata type " + std::to_string(type));
    }
}

template<class T, class F> std::vector<T> read_array(uint64_t count, F read) {
    std::vector<T> out;
    out.reserve(static_cast<size_t>(count));
    for (uint64_t i = 0; i < count; ++i) out.push_back(read());
    return out;
}
GGUFValue read_value(Reader& r, uint32_t type) {
    GGUFValue v; v.type = type;
    if (type == 9) {
        v.element_type = static_cast<uint32_t>(r.u(4));
        if (v.element_type == 9) throw std::runtime_error("nested GGUF arrays are not supported");
        uint64_t width = element_bytes(v.element_type);
        const uint64_t count = r.u(8);
        if (count > (r.file_size - r.pos) / width ||
            count > (MAX_HEADER - r.pos) / width || count > MAX_ARRAY_ITEMS)
            throw std::runtime_error("GGUF array length exceeds remaining bytes or array/header cap");
        switch (v.element_type) {
            case 0: v.data = read_array<uint8_t>(count,[&]{return uint8_t(r.u(1));}); break;
            case 1: v.data = read_array<int8_t>(count,[&]{return int8_t(r.u(1));}); break;
            case 2: v.data = read_array<uint16_t>(count,[&]{return uint16_t(r.u(2));}); break;
            case 3: v.data = read_array<int16_t>(count,[&]{return int16_t(r.u(2));}); break;
            case 4: v.data = read_array<uint32_t>(count,[&]{return uint32_t(r.u(4));}); break;
            case 5: v.data = read_array<int32_t>(count,[&]{return int32_t(r.u(4));}); break;
            case 6: v.data = read_array<float>(count,[&]{return r.f32();}); break;
            case 7: v.data = read_array<uint8_t>(count,[&]{return uint8_t(r.boolean());}); break;
            case 8: v.data = read_array<std::string>(count,[&]{return r.str();}); break;
            case 10: v.data = read_array<uint64_t>(count,[&]{return r.u(8);}); break;
            case 11: v.data = read_array<int64_t>(count,[&]{return int64_t(r.u(8));}); break;
            case 12: v.data = read_array<double>(count,[&]{return r.f64();}); break;
        }
    } else {
        switch (type) {
            case 0: v.data = uint8_t(r.u(1)); break;
            case 1: v.data = int8_t(r.u(1)); break;
            case 2: v.data = uint16_t(r.u(2)); break;
            case 3: v.data = int16_t(r.u(2)); break;
            case 4: v.data = uint32_t(r.u(4)); break;
            case 5: v.data = int32_t(r.u(4)); break;
            case 6: v.data = r.f32(); break;
            case 7: v.data = r.boolean(); break;
            case 8: v.data = r.str(); break;
            case 10: v.data = r.u(8); break;
            case 11: v.data = int64_t(r.u(8)); break;
            case 12: v.data = r.f64(); break;
            default: throw std::runtime_error("unknown GGUF metadata type " + std::to_string(type));
        }
    }
    return v;
}
} // namespace

const GGUFTypeInfo* gguf_type_info(uint32_t id) {
    static const GGUFTypeInfo types[] = {
        {0,1,4,"F32"},{1,1,2,"F16"},{2,32,18,"Q4_0"},{3,32,20,"Q4_1"},
        {6,32,22,"Q5_0"},{7,32,24,"Q5_1"},{8,32,34,"Q8_0"},{9,32,36,"Q8_1"},
        {10,256,84,"Q2_K"},{11,256,110,"Q3_K"},{12,256,144,"Q4_K"},
        {13,256,176,"Q5_K"},{14,256,210,"Q6_K"},{15,256,292,"Q8_K"},
        {16,256,66,"IQ2_XXS"},{17,256,74,"IQ2_XS"},{18,256,98,"IQ3_XXS"},
        {19,256,50,"IQ1_S"},{20,32,18,"IQ4_NL"},{21,256,110,"IQ3_S"},
        {22,256,82,"IQ2_S"},{23,256,136,"IQ4_XS"},{24,1,1,"I8"},
        {25,1,2,"I16"},{26,1,4,"I32"},{27,1,8,"I64"},{28,1,8,"F64"},
        {29,256,56,"IQ1_M"},{30,1,2,"BF16"},{34,256,54,"TQ1_0"},
        {35,256,66,"TQ2_0"},{39,32,17,"MXFP4"},
        {40,64,36,"NVFP4"},{41,128,18,"Q1_0"},{42,64,18,"Q2_0"}
    };
    for (const auto& t : types) if (t.id == id) return &t;
    return nullptr;
}

uint64_t GGUFValue::array_size() const {
    if (type != 9) return 0;
    return std::visit([](const auto& x)->uint64_t {
        using X = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<X,std::vector<uint8_t>> || std::is_same_v<X,std::vector<int8_t>> ||
                      std::is_same_v<X,std::vector<uint16_t>> || std::is_same_v<X,std::vector<int16_t>> ||
                      std::is_same_v<X,std::vector<uint32_t>> || std::is_same_v<X,std::vector<int32_t>> ||
                      std::is_same_v<X,std::vector<float>> || std::is_same_v<X,std::vector<std::string>> ||
                      std::is_same_v<X,std::vector<uint64_t>> || std::is_same_v<X,std::vector<int64_t>> ||
                      std::is_same_v<X,std::vector<double>>) return x.size();
        else return 0;
    }, data);
}

GGUFFile::~GGUFFile() { close(); }
void GGUFFile::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1; size_ = data_start_ = version_ = 0; alignment_ = 32;
    metadata_.clear(); tensors_.clear(); tensor_index_.clear();
}

bool GGUFFile::open(const std::string& path, std::string& err, bool header_only) {
    close(); err.clear();
    try {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open: " + std::string(std::strerror(errno)));
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || st.st_size < 0) throw std::runtime_error("fstat failed");
        size_ = uint64_t(st.st_size);
        Reader r{fd_,size_,0,{}};
        char magic[4];for(char& c:magic)c=char(r.u(1));
        if (std::memcmp(magic,"GGUF",4)!=0) {
            if(std::memcmp(magic,"FUGG",4)==0)
                throw std::runtime_error("byte-swapped GGUF magic: big-endian GGUF not supported");
            throw std::runtime_error("bad GGUF magic (little-endian GGUF only)");
        }
        version_ = uint32_t(r.u(4));
        if (version_ != 3) {
            if (version_ == 0x03000000u || version_ == 0x02000000u || version_ == 0x01000000u)
                throw std::runtime_error("byte-swapped GGUF version: big-endian GGUF not supported");
            throw std::runtime_error("unsupported GGUF version " + std::to_string(version_) + " (requires v3)");
        }
        const uint64_t tensor_count = r.u(8), kv_count = r.u(8);
        if (tensor_count > MAX_TENSORS) throw std::runtime_error("GGUF tensor_count exceeds limit");
        if (kv_count > MAX_KV) throw std::runtime_error("GGUF kv_count exceeds limit");
        for (uint64_t i=0;i<kv_count;++i) {
            std::string key = r.str();
            if (key.empty()) throw std::runtime_error("empty GGUF metadata key");
            uint32_t type = uint32_t(r.u(4));
            GGUFValue value = read_value(r,type);
            if (!metadata_.emplace(key,std::move(value)).second)
                throw std::runtime_error("duplicate GGUF metadata key: " + key);
        }
        if (auto p=get("general.alignment")) {
            if (p->type != 4) throw std::runtime_error("general.alignment has wrong type (requires u32)");
            alignment_ = std::get<uint32_t>(p->data);
        }
        if (alignment_ == 0 || (alignment_ & (alignment_-1)) != 0)
            throw std::runtime_error("general.alignment must be a nonzero power of two");
        if (auto p=get("split.count")) {
            if (p->type != 4 && p->type != 10)
                throw std::runtime_error("split.count has wrong type");
            uint64_t n = p->type==4 ? std::get<uint32_t>(p->data) : std::get<uint64_t>(p->data);
            if (n > 1) throw std::runtime_error("split GGUF not supported yet");
        }
        tensors_.reserve(static_cast<size_t>(tensor_count));
        std::vector<uint64_t> relative;
        relative.reserve(static_cast<size_t>(tensor_count));
        for (uint64_t i=0;i<tensor_count;++i) {
            GGUFTensor t; t.name = r.str();
            if (t.name.empty()) throw std::runtime_error("empty GGUF tensor name");
            if (tensor_index_.count(t.name)) throw std::runtime_error("duplicate GGUF tensor name: " + t.name);
            t.n_dims = uint32_t(r.u(4));
            if (t.n_dims < 1 || t.n_dims > 4) throw std::runtime_error("GGUF tensor n_dims must be 1..4: " + t.name);
            uint64_t elements=1;
            for (uint32_t j=0;j<t.n_dims;++j) {
                t.ne[j] = r.u(8);
                if (!t.ne[j]) throw std::runtime_error("GGUF tensor dimension is zero: " + t.name);
                elements = checked_mul(elements,t.ne[j],"tensor element count");
            }
            t.type = uint32_t(r.u(4));
            auto* info = gguf_type_info(t.type);
            if (!info) throw std::runtime_error("unknown GGUF tensor type id " + std::to_string(t.type));
            const uint64_t rel = r.u(8);
            if (t.ne[0] % info->block_elems)
                throw std::runtime_error("GGUF ne[0] not divisible by block size: " + t.name);
            if (rel % alignment_) throw std::runtime_error("misaligned GGUF tensor offset: " + t.name);
            t.nbytes = checked_mul(elements/info->block_elems,info->block_bytes,"tensor byte size");
            t.row_bytes = checked_mul(t.ne[0]/info->block_elems,info->block_bytes,"tensor row bytes");
            t.n_rows = elements/t.ne[0];
            tensor_index_.emplace(t.name,tensors_.size());
            tensors_.push_back(std::move(t)); relative.push_back(rel);
        }
        data_start_ = checked_add(r.pos,alignment_-1,"data alignment") & ~(alignment_-1);
        if (data_start_ > size_) throw std::runtime_error("GGUF data_start past EOF");
        for (size_t i=0;i<tensors_.size();++i) {
            auto& t=tensors_[i];
            t.offset=checked_add(data_start_,relative[i],"tensor offset");
            const uint64_t end=checked_add(t.offset,t.nbytes,"tensor end");
            if (!header_only && end > size_)
                throw std::runtime_error("GGUF tensor data past EOF: " + t.name);
        }
        std::vector<const GGUFTensor*> sorted;
        for (const auto& t:tensors_) sorted.push_back(&t);
        std::sort(sorted.begin(),sorted.end(),[](auto* a,auto* b){return a->offset<b->offset;});
        for (size_t i=1;i<sorted.size();++i)
            if (sorted[i]->offset < sorted[i-1]->offset + sorted[i-1]->nbytes)
                throw std::runtime_error("overlapping GGUF tensors: " + sorted[i-1]->name + ", " + sorted[i]->name);
        return true;
    } catch (const std::exception& e) {
        err = path + ": " + e.what(); close(); return false;
    }
}

const GGUFValue* GGUFFile::get(const std::string& key) const {
    auto it=metadata_.find(key); return it==metadata_.end()?nullptr:&it->second;
}
#define GETTER(name,typ,tag) \
bool GGUFFile::name(const std::string& key, typ& out, std::string& err) const { \
    auto* p=get(key); if (!p) {err="missing GGUF metadata: "+key;return false;} \
    if (p->type!=tag) {err="wrong GGUF metadata type: "+key;return false;} \
    out=std::get<typ>(p->data);return true; }
GETTER(get_u32,uint32_t,4)
GETTER(get_u64,uint64_t,10)
GETTER(get_f32,float,6)
GETTER(get_string,std::string,8)
#undef GETTER
bool GGUFFile::get_array(const std::string& key,const GGUFValue*& out,std::string& err) const {
    out=get(key); if (!out) {err="missing GGUF metadata: "+key;return false;}
    if (out->type!=9) {out=nullptr;err="wrong GGUF metadata type: "+key;return false;}
    return true;
}
const GGUFTensor* GGUFFile::find(const std::string& name) const {
    auto it=tensor_index_.find(name);return it==tensor_index_.end()?nullptr:&tensors_[it->second];
}
bool GGUFFile::read_raw(const GGUFTensor& t,void* dst,std::string& err) const {
    if (fd_<0 || t.offset > size_ || t.nbytes > size_-t.offset) {
        err="tensor data past EOF or GGUF file closed: "+t.name;return false;
    }
    auto* p=static_cast<uint8_t*>(dst); uint64_t done=0;
    while (done<t.nbytes) {
        size_t chunk=static_cast<size_t>(std::min<uint64_t>(t.nbytes-done,1u<<20));
        ssize_t n=::pread(fd_,p+done,chunk,static_cast<off_t>(t.offset+done));
        if (n<0 && errno==EINTR) continue;
        if (n<=0) {err="pread failed for GGUF tensor: "+t.name;return false;}
        done+=static_cast<uint64_t>(n);
    }
    return true;
}
bool GGUFFile::read_raw_row(const GGUFTensor& t,uint64_t row,void* dst,std::string& err) const {
    if (t.row_bytes==0 || row>=t.n_rows || t.row_bytes > t.nbytes ||
        row > (t.nbytes-t.row_bytes)/t.row_bytes) {
        err="GGUF row out of range: "+t.name;return false;
    }
    GGUFTensor one=t;
    one.offset=t.offset+row*t.row_bytes;
    one.nbytes=t.row_bytes;
    return read_raw(one,dst,err);
}
} // namespace b70
