#pragma once
// Torch-free subset of Intel's Xe2 FMHA policy declarations.  Keeping this
// local makes the production bridge a raw SYCL/Level Zero library.
#include <cute/tensor.hpp>

#define HEAD_SIZE_LIMIT_0 64
#define HEAD_SIZE_LIMIT_1 96
#define HEAD_SIZE_LIMIT_2 128
#define HEAD_SIZE_LIMIT_3 192
#define HEAD_SIZE_LIMIT_4 256
#define HEAD_SIZE_LIMIT_5 512

enum class CutlassDType { half, bfloat16, float8_e4m3, float8_e5m2 };
struct CutlassQKType {
  CutlassDType q_type, k_type;
  explicit CutlassQKType(CutlassDType t):q_type(t),k_type(t){}
  CutlassQKType(CutlassDType q,CutlassDType k):q_type(q),k_type(k){}
};

using namespace cute;
#define GRIMOIRE_CHUNK_POLICY(NAME, M, O, SG) \
struct NAME { using ShapeQK=Shape<M,_32,_32>; using ShapePV=Shape<M,_32,_32>; \
  using ShapeOut=Shape<M,O>; using SubgroupLayoutQK=Layout<Shape<SG,_1,_1>>; };
GRIMOIRE_CHUNK_POLICY(chunk_policy_head64,  _128, _64,  _8)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head96,  _128, _96,  _8)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head128, _256, _128, _16)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head192, _256, _192, _32)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head256, _256, _256, _32)
GRIMOIRE_CHUNK_POLICY(chunk_policy_head512, _256, _256, _32)
#undef GRIMOIRE_CHUNK_POLICY
