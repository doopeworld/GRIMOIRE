#include "b70/native_model.hpp"
#include "b70/safetensors.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace b70 {
namespace {
size_t dtype_size(uint32_t t) {
    switch (static_cast<STDtype>(t)) {
    case STDtype::F64: case STDtype::I64: return 8;
    case STDtype::F32: case STDtype::I32: return 4;
    case STDtype::F16: case STDtype::BF16: case STDtype::I16: return 2;
    case STDtype::F8_E4M3: case STDtype::F8_E5M2:
    case STDtype::I8: case STDtype::U8: case STDtype::BOOL: return 1;
    default: return 0;
    }
}
bool elements(const NativeTensorRecord& r, uint64_t& n) {
    if (r.rank > 4) return false;
    n = 1;
    for (uint32_t d = 0; d < r.rank; ++d) {
        if (r.shape[d] <= 0 || n > UINT64_MAX / uint64_t(r.shape[d])) return false;
        n *= uint64_t(r.shape[d]);
    }
    return true;
}
}
size_t scale_element_bytes(Fmt f) {
    switch (f) {
    case Fmt::BF16: return 0;
    case Fmt::FP8_E4M3: case Fmt::FP8_E5M2: case Fmt::INT8: return 4;
    case Fmt::INT4: return 2;
    case Fmt::MXFP8: case Fmt::MXFP4: return 1;
    }
    throw std::invalid_argument("unknown weight format");
}
NativeEncoding native_encoding(Fmt f) {
    switch (f) {
    case Fmt::BF16: return NativeEncoding::RAW;
    case Fmt::MXFP4: return NativeEncoding::MXFP4_GRIMOIRE_XE2;
    case Fmt::MXFP8: return NativeEncoding::MXFP8_GRIMOIRE_XE2;
    case Fmt::INT4: return NativeEncoding::INT4_AFFINE_G128;
    case Fmt::FP8_E4M3: return NativeEncoding::FP8_E4M3_CHANNEL;
    case Fmt::FP8_E5M2: return NativeEncoding::FP8_E5M2_CHANNEL;
    case Fmt::INT8: return NativeEncoding::INT8_CHANNEL;
    }
    throw std::invalid_argument("unknown weight format");
}
const char* native_encoding_name(uint32_t e) {
    switch (static_cast<NativeEncoding>(e)) {
    case NativeEncoding::RAW: return "raw";
    case NativeEncoding::MXFP4_GRIMOIRE_XE2: return "mxfp4";
    case NativeEncoding::MXFP8_GRIMOIRE_XE2: return "mxfp8";
    case NativeEncoding::INT4_AFFINE_G128: return "int4-g128";
    case NativeEncoding::FP8_E4M3_CHANNEL: return "fp8-e4m3";
    case NativeEncoding::FP8_E5M2_CHANNEL: return "fp8-e5m2";
    case NativeEncoding::INT8_CHANNEL: return "int8";
    }
    return "unknown";
}
bool native_layout(const NativeTensorRecord& r, NativeLayout& out, std::string& err) {
    out = {};
    uint64_t n;
    if (!elements(r, n) || r.rank < 2 || r.shape[r.rank-1] > INT32_MAX/2 ||
        n / uint64_t(r.shape[r.rank-1]) > INT32_MAX) {
        err = "invalid native matrix dimensions"; return false;
    }
    out.K = int(r.shape[r.rank-1]); out.N = int(n / out.K);
    switch (static_cast<NativeEncoding>(r.encoding)) {
    case NativeEncoding::RAW:
        if (r.source_dtype != uint32_t(STDtype::BF16)) {
            err = "raw matrix is not BF16"; return false;
        }
        out.fmt = Fmt::BF16; break;
    case NativeEncoding::MXFP4_GRIMOIRE_XE2: out.fmt = Fmt::MXFP4; break;
    case NativeEncoding::MXFP8_GRIMOIRE_XE2: out.fmt = Fmt::MXFP8; break;
    case NativeEncoding::INT4_AFFINE_G128: out.fmt = Fmt::INT4; break;
    case NativeEncoding::FP8_E4M3_CHANNEL: out.fmt = Fmt::FP8_E4M3; break;
    case NativeEncoding::FP8_E5M2_CHANNEL: out.fmt = Fmt::FP8_E5M2; break;
    case NativeEncoding::INT8_CHANNEL: out.fmt = Fmt::INT8; break;
    default: err = "unknown native encoding"; return false;
    }
    const int group = out.fmt == Fmt::INT4 ? kInt4Group :
        (out.fmt == Fmt::MXFP4 || out.fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (out.K % group) { err = "incomplete native quantization group"; return false; }
    out.row_payload_bytes = uint64_t(bytes_per_row(out.fmt, out.K));
    out.row_scale_bytes = uint64_t(scales_per_row(out.fmt, out.K)) * scale_element_bytes(out.fmt);
    out.row_zero_bytes = out.fmt == Fmt::INT4 ? uint64_t(out.K / kInt4Group) : 0;
    out.payload_bytes = uint64_t(out.N) * out.row_payload_bytes;
    out.scale_bytes = uint64_t(out.N) * out.row_scale_bytes;
    out.zero_bytes = uint64_t(out.N) * out.row_zero_bytes;
    if (r.payload_bytes != out.payload_bytes || r.scales_bytes != out.scale_bytes + out.zero_bytes) {
        err = "native payload/scale sizes do not match encoding and shape"; return false;
    }
    return true;
}
QuantWeight slice_quant_rows(const QuantWeight& w,int first,int rows) {
    if (first<0 || rows<=0 || first>w.N || rows>w.N-first)
        throw std::invalid_argument("quantized row slice out of bounds");
    QuantWeight out=w;out.N=rows;
    out.payload=w.payload+int64_t(first)*w.row_bytes;
    if (w.scales) out.scales=static_cast<const uint8_t*>(w.scales)+
        int64_t(first)*w.row_scales*scale_element_bytes(w.fmt);
    if (w.zeros) out.zeros=w.zeros+int64_t(first)*w.row_scales;
    return out;
}
PackedWeight copy_packed(const QuantWeight& w) {
    PackedWeight p;
    p.fmt = w.fmt; p.N = w.N; p.K = w.K;
    p.row_bytes = w.row_bytes; p.row_scales = w.row_scales;
    p.payload.assign(w.payload, w.payload + size_t(w.N) * w.row_bytes);
    const size_t sb = size_t(w.N) * w.row_scales * scale_element_bytes(w.fmt);
    if (sb) {
        const auto* s = static_cast<const uint8_t*>(w.scales);
        p.scales_raw.assign(s, s + sb);
    }
    if (w.fmt == Fmt::INT4) p.zeros.assign(w.zeros, w.zeros + size_t(w.N) * w.row_scales);
    return p;
}
NativeModel::~NativeModel(){close();}
void NativeModel::close(){
    index_.clear();
    if(base_){::munmap(base_,mapped_size_);base_=nullptr;}
    if(fd_>=0){::close(fd_);fd_=-1;}
    mapped_size_=0;header_={};toc_=nullptr;
}
void NativeModel::drop_resident(){ if(base_) ::madvise(base_, mapped_size_, MADV_DONTNEED); }
bool NativeModel::open(const std::string& path,std::string& err){
    close(); err.clear(); fd_=::open(path.c_str(),O_RDONLY);
    if(fd_<0){err="cannot open "+path;return false;}
    auto fail = [&](std::string why) { err=std::move(why); close(); return false; };
    struct stat st{};
    if(::fstat(fd_,&st) || st.st_size < 0) return fail("fstat failed");
    mapped_size_=uint64_t(st.st_size);
    if(mapped_size_<sizeof(header_) || mapped_size_>SIZE_MAX) return fail("truncated or oversized native model");
    base_=::mmap(nullptr,size_t(mapped_size_),PROT_READ,MAP_PRIVATE,fd_,0);
    if(base_==MAP_FAILED){base_=nullptr;return fail("mmap failed");}
    std::memcpy(&header_,base_,sizeof header_);
    if(header_.magic!=kNativeMagic || (header_.version!=2 && header_.version!=kNativeVersion) ||
       header_.alignment!=kNativeAlignment || header_.file_size!=mapped_size_ ||
       header_.target!=0x031 || header_.reserved!=0) return fail("invalid or unsupported B70 native header");
    if(header_.toc_offset<kNativeAlignment || header_.toc_offset>mapped_size_ ||
       header_.toc_offset%kNativeAlignment || header_.tensor_count==0 ||
       header_.tensor_count>(mapped_size_-header_.toc_offset)/sizeof(NativeTensorRecord) ||
       header_.tensor_count*sizeof(NativeTensorRecord)!=mapped_size_-header_.toc_offset)
        return fail("invalid native model TOC");
    toc_=reinterpret_cast<const NativeTensorRecord*>(static_cast<const uint8_t*>(base_)+header_.toc_offset);
    std::vector<std::pair<uint64_t,uint64_t>> ranges;
    auto range = [&](uint64_t off, uint64_t bytes) {
        if (!bytes) return off==0;
        if (off<kNativeAlignment || off%kNativeAlignment || off>header_.toc_offset || bytes>header_.toc_offset-off) return false;
        ranges.emplace_back(off, off+bytes); return true;
    };
    for(uint64_t i=0;i<header_.tensor_count;++i){
        const auto& r=toc_[i];
        const size_t len=strnlen(r.name,sizeof r.name);
        if (!len || len==sizeof r.name || !index_.emplace(std::string(r.name,len),&r).second)
            return fail("invalid or duplicate native tensor name");
        uint64_t n;
        const size_t ds=dtype_size(r.source_dtype);
        if (!ds || !elements(r,n) || n>UINT64_MAX/ds || r.flags)
            return fail("invalid native tensor shape, dtype or flags: "+std::string(r.name));
        if (!range(r.payload_offset,r.payload_bytes) || !range(r.scales_offset,r.scales_bytes))
            return fail("native tensor outside aligned data region: "+std::string(r.name));
        if (r.encoding==uint32_t(NativeEncoding::RAW)) {
            if (r.payload_bytes!=n*ds || r.scales_bytes || r.padded_n || r.padded_k || r.tile_n || r.tile_k)
                return fail("invalid raw native tensor: "+std::string(r.name));
        } else {
            NativeLayout l;
            if ((header_.version==2 && r.encoding!=1) || r.source_dtype!=uint32_t(STDtype::BF16) ||
                !native_layout(r,l,err)) return fail("unsupported native matrix: "+std::string(r.name)+" "+err);
            if (r.padded_n!=l.N || r.padded_k!=l.K || r.tile_n!=64 || r.tile_k!=256)
                return fail("unsupported native matrix layout: "+std::string(r.name));
            const auto* s=static_cast<const uint8_t*>(scales(r));
            const size_t unit=scale_element_bytes(l.fmt);
            for (uint64_t j=0;j<l.scale_bytes;j+=unit) {
                float scale;
                if (unit==4) std::memcpy(&scale,s+j,4);
                else if (unit==2) { bf16_t b; std::memcpy(&b,s+j,2); scale=bf16_to_f32(b); }
                else scale=e8m0_to_f32(s[j]);
                if (!(scale>0) || !std::isfinite(scale)) return fail("non-positive or non-finite native scale: "+std::string(r.name));
            }
            for (uint64_t j=0;j<l.zero_bytes;++j)
                if (s[l.scale_bytes+j]>15) return fail("invalid native INT4 zero point: "+std::string(r.name));
        }
    }
    std::sort(ranges.begin(),ranges.end());
    for (size_t i=1;i<ranges.size();++i)
        if (ranges[i].first<ranges[i-1].second) return fail("overlapping native tensor regions");
    return true;
}
const NativeTensorRecord* NativeModel::find(const std::string& n)const{
    auto it=index_.find(n);return it==index_.end()?nullptr:it->second;
}
const void* NativeModel::payload(const NativeTensorRecord& r)const{
    return static_cast<const uint8_t*>(base_)+r.payload_offset;
}
const void* NativeModel::scales(const NativeTensorRecord& r)const{
    return r.scales_bytes?static_cast<const uint8_t*>(base_)+r.scales_offset:nullptr;
}
bool NativeModel::view(const NativeTensorRecord& r,uint64_t first,int rows,QuantWeight& out,std::string& err)const{
    out={}; NativeLayout l;
    if (!base_ || !native_layout(r,l,err)) return false;
    if (rows<=0 || first>uint64_t(l.N) || uint64_t(rows)>uint64_t(l.N)-first) {
        err="native row slice out of bounds"; return false;
    }
    out.fmt=l.fmt; out.N=rows; out.K=l.K;
    out.payload=static_cast<const uint8_t*>(payload(r))+first*l.row_payload_bytes;
    const auto* s=static_cast<const uint8_t*>(scales(r));
    out.scales=l.scale_bytes?s+first*l.row_scale_bytes:nullptr;
    out.zeros=l.zero_bytes?s+l.scale_bytes+first*l.row_zero_bytes:nullptr;
    out.row_bytes=int64_t(l.row_payload_bytes); out.row_scales=scales_per_row(l.fmt,l.K);
    return true;
}
} // namespace b70
