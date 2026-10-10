// GRIMOIRE — Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
#include "kernels.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <stdexcept>
using namespace b70;
namespace sx=sycl::ext::oneapi::experimental;
static uint32_t bits(float v){uint32_t b;std::memcpy(&b,&v,4);return b;}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 std::ifstream input(argv[1]);if(!input)return 2;
 std::vector<int8_t> base_codes;std::vector<bf16_t> scales;std::vector<float> expected;
 int code,half,value;
 while(input>>code>>half>>value){
  if(base_codes.size()%256==0) scales.push_back(f32_to_bf16(f16_to_f32(uint16_t(half))));
  base_codes.push_back(int8_t(code));expected.push_back(bf16_to_f32(bf16_t{uint16_t(value)}));
 }
 const int rows=int(scales.size());if(rows<2||base_codes.size()!=size_t(rows)*256)return 2;
 sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
 std::printf("device=%s oracle_rows=%d\n",q.get_device().get_info<sycl::info::device::name>().c_str(),rows);
 for(int H:{256,5120}){
  std::vector<int8_t> codes(size_t(rows)*H);
  for(int r=0;r<rows;++r)for(int d=0;d<H;++d)codes[size_t(r)*H+d]=base_codes[size_t(r)*256+d%256];
  auto* table=sycl::malloc_device<int8_t>(codes.size(),q);
  auto* ds=sycl::malloc_device<bf16_t>(scales.size(),q);
  auto* tokens=sycl::malloc_device<int32_t>(rows,q);
  auto* out=sycl::malloc_device<float>(size_t(rows)*H,q);
  if(!table||!ds||!tokens||!out)throw std::runtime_error("allocation failed");
  q.memcpy(table,codes.data(),codes.size());q.memcpy(ds,scales.data(),scales.size()*sizeof(bf16_t));
  std::vector<int32_t> ids(rows);for(int r=0;r<rows;++r)ids[r]=r;
  q.memcpy(tokens,ids.data(),rows*sizeof(int32_t));q.wait_and_throw();
  auto verify=[&](const std::vector<int32_t>& chosen,int count,int begin,int owned){
   std::vector<float> got(size_t(count)*H);q.memcpy(got.data(),out,got.size()*4).wait();
   for(int r=0;r<count;++r)for(int d=0;d<H;++d){
    const int id=chosen[size_t(r)];
    const float want=id>=begin&&id<begin+owned?expected[size_t(id)*256+d%256]:0.0f;
    if(bits(got[size_t(r)*H+d])!=bits(want)){
     std::printf("FAIL H=%d row=%d token=%d dim=%d got=%08x expected=%08x\n",H,r,id,d,bits(got[size_t(r)*H+d]),bits(want));throw std::runtime_error("oracle mismatch");
    }
   }
  };
  launch_embed_int8_batched(q,table,ds,tokens,out,rows,H,0,rows,{});q.wait_and_throw();verify(ids,rows,0,rows);
  launch_embed_int8(q,table,ds,rows-1,out,H,{});q.wait_and_throw();verify({rows-1},1,0,rows);
  const int begin=rows/2,owned=rows-begin;
  launch_embed_int8_batched(q,table+size_t(begin)*H,ds+begin,tokens,out,rows,H,begin,owned,{});q.wait_and_throw();verify(ids,rows,begin,owned);
  sx::command_graph<sx::graph_state::modifiable> graph(q.get_context(),q.get_device());
  graph.begin_recording(q);launch_embed_int8_batched(q,table,ds,tokens,out,rows,H,0,rows,{});graph.end_recording(q);
  auto executable=graph.finalize();for(int r=0;r<rows;++r)ids[r]=rows-1-r;
  q.memcpy(tokens,ids.data(),rows*sizeof(int32_t)).wait();q.ext_oneapi_graph(executable).wait();verify(ids,rows,0,rows);
  std::printf("PASS H=%d single/batch/shard/dynamic-graph exact BF16 oracle\n",H);
  sycl::free(table,q);sycl::free(ds,q);sycl::free(tokens,q);sycl::free(out,q);
 }
}
