#ifndef MSL_PRELUDE
#define MSL_PRELUDE

#include <metal_stdlib>
using namespace metal;

// Scalar-block-layout matrices. MSL pads each column of float3x3 and float4x4 to its vector
// alignment, so a matrix that must match the CPU byte for byte carries packed columns and
// converts to the arithmetic type on use.
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

// A bindless sampler slot carries the texture and the sampler side by side.
struct BindlessSampler2D {
    texture2d<float> Texture;
    sampler Sampler;
};
struct BindlessSamplerCube {
    texturecube<float> Texture;
    sampler Sampler;
};

#endif
