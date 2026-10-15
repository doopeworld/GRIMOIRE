// Minimal GGUF v3 writer shared by parser and qwen35 adapter fixtures.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

namespace gguf_test_writer {
using Bytes=std::vector<uint8_t>;
inline void put(Bytes& b,uint64_t v,unsigned n) {for(unsigned i=0;i<n;++i)b.push_back(uint8_t(v>>(8*i)));}
inline void str(Bytes& b,const std::string& s) {put(b,s.size(),8);b.insert(b.end(),s.begin(),s.end());}
inline Bytes num(uint64_t v,unsigned n) {Bytes b;put(b,v,n);return b;}
struct KV {std::string key;uint32_t type;Bytes payload;};
struct TS {std::string name;std::vector<uint64_t> ne;uint32_t type;uint64_t offset;Bytes raw;};
inline KV kv_string(std::string k,std::string s) {Bytes b;str(b,s);return {k,8,b};}
inline KV kv_array_strings(std::string k,std::vector<std::string> items) {
    Bytes b;put(b,8,4);put(b,items.size(),8);for(auto& s:items)str(b,s);return {k,9,b};
}
inline Bytes build_file(const std::vector<KV>& kvs,const std::vector<TS>& ts,uint32_t align=64) {
    Bytes b={'G','G','U','F'};put(b,3,4);put(b,ts.size(),8);put(b,kvs.size(),8);
    for(const auto& k:kvs){str(b,k.key);put(b,k.type,4);b.insert(b.end(),k.payload.begin(),k.payload.end());}
    for(const auto& t:ts){str(b,t.name);put(b,t.ne.size(),4);for(auto n:t.ne)put(b,n,8);put(b,t.type,4);put(b,t.offset,8);}
    b.resize((b.size()+align-1)&~size_t(align-1),0);
    size_t base=b.size();
    for(const auto& t:ts) {
        if(t.offset+t.raw.size()>SIZE_MAX-base) continue;
        size_t end=base+static_cast<size_t>(t.offset+t.raw.size());if(end>b.size())b.resize(end,0);
        std::copy(t.raw.begin(),t.raw.end(),b.begin()+base+static_cast<size_t>(t.offset));
    }
    return b;
}
}
