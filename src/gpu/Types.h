#pragma once

// GPU records use matching C++ numeric and native packed Metal storage.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

namespace numeric {
using vec2 = metal::packed_float2;
using vec3 = metal::packed_float3;
using vec4 = metal::packed_float4;
using uvec2 = metal::packed_uint2;
using uvec3 = metal::packed_uint3;
using uvec4 = metal::packed_uint4;
using quat = metal::packed_float4;

// Packed columns preserve the C++ matrix buffer layout.
struct mat3 {
    vec3 Columns[3];
    metal::float3x3 Unpack() const constant { return metal::float3x3(metal::float3(Columns[0]), metal::float3(Columns[1]), metal::float3(Columns[2])); }
    metal::float3x3 Unpack() const device { return metal::float3x3(metal::float3(Columns[0]), metal::float3(Columns[1]), metal::float3(Columns[2])); }
    metal::float3x3 Unpack() const thread { return metal::float3x3(metal::float3(Columns[0]), metal::float3(Columns[1]), metal::float3(Columns[2])); }
};
struct mat4 {
    vec4 Columns[4];
    metal::float4x4 Unpack() const constant { return metal::float4x4(metal::float4(Columns[0]), metal::float4(Columns[1]), metal::float4(Columns[2]), metal::float4(Columns[3])); }
    metal::float4x4 Unpack() const device { return metal::float4x4(metal::float4(Columns[0]), metal::float4(Columns[1]), metal::float4(Columns[2]), metal::float4(Columns[3])); }
    metal::float4x4 Unpack() const thread { return metal::float4x4(metal::float4(Columns[0]), metal::float4(Columns[1]), metal::float4(Columns[2]), metal::float4(Columns[3])); }
};
} // namespace numeric

#define GPU_CONSTANT constant constexpr
// Field defaults initialize CPU records only.
#define DEFAULT(...)

template<typename T, size_t N> using GpuArray = metal::array<T, N>;
#else
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/uvec2.h"
#include "numeric/uvec3.h"
#include "numeric/uvec4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <array>
#include <cstdint>

#define GPU_CONSTANT constexpr
#define DEFAULT(...) {__VA_ARGS__}

template<typename T, std::size_t N> using GpuArray = std::array<T, N>;
#endif

using numeric::mat3, numeric::mat4, numeric::quat, numeric::uvec2, numeric::uvec3, numeric::uvec4, numeric::vec2, numeric::vec3, numeric::vec4;
static_assert(sizeof(vec2) == 8 && alignof(vec2) == 4);
static_assert(sizeof(vec3) == 12 && alignof(vec3) == 4);
static_assert(sizeof(vec4) == 16 && alignof(vec4) == 4);
static_assert(sizeof(uvec2) == 8 && alignof(uvec2) == 4);
static_assert(sizeof(uvec3) == 12 && alignof(uvec3) == 4);
static_assert(sizeof(uvec4) == 16 && alignof(uvec4) == 4);
static_assert(sizeof(quat) == 16 && alignof(quat) == 4);
static_assert(sizeof(mat3) == 36 && alignof(mat3) == 4);
static_assert(sizeof(mat4) == 64 && alignof(mat4) == 4);

GPU_CONSTANT uint32_t InvalidSlot = ~0u;
GPU_CONSTANT uint32_t InvalidOffset = ~0u;
