// GRIMOIRE — Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/int8_embedding.hpp"
#include "b70/qwen35.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
using namespace b70;
static void check(bool yes,const char* what){if(!yes){std::fprintf(stderr,"FAIL: %s\n",what);std::exit(1);}}
int main(int argc,char**argv){
 check(argc>=2,"fixture directory required"); const std::string root=argv[1];
 std::ifstream f(root+"/oracle.txt"); check(bool(f),"oracle input missing");
 int code,half,expected; size_t tested=0,bad_unscaled=0,bad_late_scale=0;
 while(f>>code>>half>>expected){
  const float scale=f16_to_f32(uint16_t(half));
  const float value=int8_embedding_value(int8_t(code),f32_to_bf16(scale));
  if(f32_to_bf16(value).bits!=expected) std::fprintf(stderr,"code=%d scale=%04x actual=%04x expected=%04x\n",code,half,f32_to_bf16(value).bits,expected);
  check(f32_to_bf16(value).bits==expected,"lookup disagrees with rational BF16 oracle");
  bad_unscaled+=f32_to_bf16(float(code)).bits!=expected;
  bad_late_scale+=f32_to_bf16(float(code)*scale).bits!=expected;
  ++tested;
 }
 check(tested>40000,"oracle coverage missing");
 check(bad_unscaled && bad_late_scale,"negative controls did not exercise scaling/rounding");
 for(const char* name:{"early","late","dense","missing","bits","packed","path","shape","dtype","negative","nan","unscaled"}){
  Qwen35Model m;std::string err;const bool ok=m.load(root+"/"+name,err);
  const bool valid=std::string(name)=="early"||std::string(name)=="late"||std::string(name)=="dense";
  check(ok==valid,(std::string(name)+": "+err).c_str());
  if(ok && std::string(name)!="dense"){
   check(m.embed.int8_embedding&&m.embed.t.dtype==STDtype::I8,"side-file selection depends on ordering");
   check(m.embed.scales_shard==m.embed.shard&&m.embed.scales_t.dtype==STDtype::F16,"scale reference lost");
  }else if(ok) check(!m.embed.int8_embedding&&m.embed.t.dtype==STDtype::BF16,"dense embedding changed");
  std::printf("loader %s: %s %s\n",name,ok?"accepted":"rejected",err.c_str());
 }
 if(argc>2){Qwen35Model m;std::string err;check(m.load(argv[2],err),err.c_str());
  check(m.embed.int8_embedding&&m.embed.t.dtype==STDtype::I8,"real Swift embedding not selected");
  check(m.embed.scales_shard==m.embed.shard,"real Swift scales not attached");
  std::printf("Swift: resident-source I8 [%lld,%lld], scales %s; lm_head GPTQ=%d\n",
   (long long)m.embed.t.shape[0],(long long)m.embed.t.shape[1],st_dtype_name(m.embed.scales_t.dtype),m.lm_head.gptq);
  check(m.lm_head.gptq,"baked output head did not resolve as GPTQ");
  for(const char*name:{"mtp.fc","mtp.layers.0.self_attn.q_proj","mtp.layers.0.self_attn.k_proj","mtp.layers.0.self_attn.v_proj","mtp.layers.0.self_attn.o_proj","mtp.layers.0.mlp.gate_proj","mtp.layers.0.mlp.up_proj","mtp.layers.0.mlp.down_proj"}) check(m.linear_ref(name).gptq,name);
 }
 std::printf("PASS %zu oracle values; negative mismatches raw=%zu late-scale=%zu\n",tested,bad_unscaled,bad_late_scale);
}
