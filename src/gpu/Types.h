#pragma once

// Type aliases shared by the C++ and MSL sides of the GPU headers.
// Vectors and matrices are scalar-packed on both sides, so one struct definition has one layout.

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;

#define GPU_CONSTANT constant constexpr
// Field defaults initialize CPU records only.
#define DEFAULT(...)

template<typename T, size_t N> using GpuArray = metal::array<T, N>;
using vec2 = packed_float2;
using vec3 = packed_float3;
using vec4 = packed_float4;
using uvec2 = packed_uint2;
using uvec4 = packed_uint4;
using quat = packed_float4;

// Packed columns preserve CPU scalar-block matrix layout despite MSL vector-aligned matrix columns.
struct packed_float3x3 {
    packed_float3 Columns[3];
    float3x3 Unpack() const constant { return float3x3(float3(Columns[0]), float3(Columns[1]), float3(Columns[2])); }
    float3x3 Unpack() const device { return float3x3(float3(Columns[0]), float3(Columns[1]), float3(Columns[2])); }
    float3x3 Unpack() const thread { return float3x3(float3(Columns[0]), float3(Columns[1]), float3(Columns[2])); }
};
struct packed_float4x4 {
    packed_float4 Columns[4];
    float4x4 Unpack() const constant { return float4x4(float4(Columns[0]), float4(Columns[1]), float4(Columns[2]), float4(Columns[3])); }
    float4x4 Unpack() const device { return float4x4(float4(Columns[0]), float4(Columns[1]), float4(Columns[2]), float4(Columns[3])); }
    float4x4 Unpack() const thread { return float4x4(float4(Columns[0]), float4(Columns[1]), float4(Columns[2]), float4(Columns[3])); }
};
using mat3 = packed_float3x3;
using mat4 = packed_float4x4;
#else
#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <array>
#include <cstdint>

#define GPU_CONSTANT constexpr
#define DEFAULT(...) {__VA_ARGS__}

template<typename T, std::size_t N> using GpuArray = std::array<T, N>;
#endif

GPU_CONSTANT uint32_t InvalidSlot = ~0u;
GPU_CONSTANT uint32_t InvalidOffset = ~0u;
