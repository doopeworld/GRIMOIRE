// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/native_model.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

namespace b70 {
NativeModel::~NativeModel(){close();}
void NativeModel::close(){
    index_.clear();
    if(base_){::munmap(base_,mapped_size_);base_=nullptr;}
    if(fd_>=0){::close(fd_);fd_=-1;}
    mapped_size_=0;header_={};toc_=nullptr;
}
bool NativeModel::open(const std::string& path,std::string& err){
    close(); fd_=::open(path.c_str(),O_RDONLY);
    if(fd_<0){err="cannot open "+path;return false;}
    struct stat st{}; if(::fstat(fd_,&st)){err="fstat failed";close();return false;}
    mapped_size_=uint64_t(st.st_size);
    if(mapped_size_<sizeof(header_)){err="truncated native model";close();return false;}
    base_=::mmap(nullptr,mapped_size_,PROT_READ,MAP_PRIVATE,fd_,0);
    if(base_==MAP_FAILED){base_=nullptr;err="mmap failed";close();return false;}
    std::memcpy(&header_,base_,sizeof header_);
    if(header_.magic!=kNativeMagic||header_.version!=kNativeVersion||
       header_.alignment!=kNativeAlignment||header_.file_size!=mapped_size_||
       header_.target!=0x031){err="invalid or non-B70 native model";close();return false;}
    if(header_.toc_offset>mapped_size_||header_.tensor_count>
       (mapped_size_-header_.toc_offset)/sizeof(NativeTensorRecord)){
        err="invalid native model TOC";close();return false;
    }
    auto* toc=reinterpret_cast<const NativeTensorRecord*>(
        static_cast<const uint8_t*>(base_)+header_.toc_offset);
    toc_=toc;
    for(uint64_t i=0;i<header_.tensor_count;++i){
        const auto& r=toc[i];
        if(r.payload_offset+r.payload_bytes>header_.toc_offset||
           (r.scales_bytes&&r.scales_offset+r.scales_bytes>header_.toc_offset)){
            err="native tensor outside data region";close();return false;
        }
        index_.emplace(std::string(r.name,strnlen(r.name,sizeof r.name)),&r);
    }
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
} // namespace b70
