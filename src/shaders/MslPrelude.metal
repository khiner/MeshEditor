#ifndef MSL_PRELUDE
#define MSL_PRELUDE

#include <metal_stdlib>
using namespace metal;

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

struct BindlessSampler2D {
    texture2d<float> Texture;
    sampler Sampler;
};
struct BindlessSamplerCube {
    texturecube<float> Texture;
    sampler Sampler;
};

#endif
