#ifndef IBLPREFILTER_MSL
#define IBLPREFILTER_MSL

// Environment prefilter kernels bind source and destination textures directly.
#include <metal_stdlib>
#include "CubemapFace.metal"
using namespace metal;

constant float PI = 3.14159265358979323846f;

struct CubeFacePushConstants {
    uint FaceSize;
};

struct PrefilterPushConstants {
    uint FaceSize;
    uint SourceSize;
    float Roughness;
};

kernel void EquirectToCubemapKernel(
    uint3 gid [[thread_position_in_grid]],
    texture2d<float> equirect_map [[texture(0)]],
    sampler equirect_sampler [[sampler(0)]],
    texture2d_array<float, access::write> cube_face [[texture(1)]],
    constant CubeFacePushConstants &pc [[buffer(0)]]
) {
    const uint face = gid.z;
    const int2 px = int2(gid.xy);
    if (px.x >= int(pc.FaceSize) || px.y >= int(pc.FaceSize)) return;

    const float2 uv = (float2(px) + 0.5f) / float(pc.FaceSize) * 2.0f - 1.0f;
    const float3 dir = FaceDirection(face, uv);

    const float lon = atan2(dir.z, dir.x) / (2.0f * PI) + 0.5f;
    const float lat = asin(clamp(dir.y, -1.0f, 1.0f)) / PI + 0.5f;

    // Flip V: lat 0 is the south pole, but equirect images put V=0 at the top (north).
    const float4 color = equirect_map.sample(equirect_sampler, float2(lon, 1.0f - lat), level(0.0f));
    cube_face.write(color, uint2(px), face);
}

kernel void DiffuseIrradianceKernel(
    uint3 gid [[thread_position_in_grid]],
    texturecube<float> env_map [[texture(0)]],
    sampler env_sampler [[sampler(0)]],
    texture2d_array<float, access::write> irradiance_face [[texture(1)]],
    constant CubeFacePushConstants &pc [[buffer(0)]]
) {
    const uint face = gid.z;
    const int2 px = int2(gid.xy);
    if (px.x >= int(pc.FaceSize) || px.y >= int(pc.FaceSize)) return;

    const float2 uv = (float2(px) + 0.5f) / float(pc.FaceSize) * 2.0f - 1.0f;
    const float3 normal = FaceDirection(face, uv);

    const float3 up = abs(normal.y) < 0.999f ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 right = normalize(cross(up, normal));
    const float3 fwd = cross(normal, right);

    float3 irradiance = float3(0.0f);
    float sample_count = 0.0f;
    const float delta = 0.025f;
    for (float phi = 0.0f; phi < 2.0f * PI; phi += delta) {
        for (float theta = 0.0f; theta < 0.5f * PI; theta += delta) {
            const float3 tangent_dir = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            const float3 sample_dir = tangent_dir.x * right + tangent_dir.y * fwd + tangent_dir.z * normal;
            irradiance += env_map.sample(env_sampler, sample_dir, level(0.0f)).rgb * cos(theta) * sin(theta);
            sample_count += 1.0f;
        }
    }
    irradiance = PI * irradiance / sample_count;
    irradiance_face.write(float4(irradiance, 1.0f), uint2(px), face);
}

inline float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

inline float2 Hammersley(uint i, uint N) { return float2(float(i) / float(N), RadicalInverse_VdC(i)); }

inline float3 ImportanceSampleGGX(float2 Xi, float3 N, float roughness) {
    const float a = roughness * roughness;
    const float phi = 2.0f * PI * Xi.x;
    const float cos_theta = sqrt((1.0f - Xi.y) / (1.0f + (a * a - 1.0f) * Xi.y));
    const float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    const float3 H = float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
    const float3 up = abs(N.z) < 0.999f ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
    const float3 tangent = normalize(cross(up, N));
    const float3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

inline float DistributionGGX(float NdotH, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(PI * denom * denom, 1e-6f);
}

kernel void SpecularPrefilterKernel(
    uint3 gid [[thread_position_in_grid]],
    texturecube<float> env_map [[texture(0)]],
    sampler env_sampler [[sampler(0)]],
    texture2d_array<float, access::write> prefilter_face [[texture(1)]],
    constant PrefilterPushConstants &pc [[buffer(0)]]
) {
    const uint face = gid.z;
    const int2 px = int2(gid.xy);
    if (px.x >= int(pc.FaceSize) || px.y >= int(pc.FaceSize)) return;

    const float2 uv = (float2(px) + 0.5f) / float(pc.FaceSize) * 2.0f - 1.0f;
    const float3 N = FaceDirection(face, uv);
    const float3 V = N;

    const uint SAMPLE_COUNT = 1024u;
    const float src_size = float(max(pc.SourceSize, 1u));
    const float max_src_lod = log2(src_size);

    float3 prefiltered = float3(0.0f);
    float total_weight = 0.0f;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        const float2 Xi = Hammersley(i, SAMPLE_COUNT);
        const float3 H = ImportanceSampleGGX(Xi, N, pc.Roughness);
        const float3 L = normalize(2.0f * dot(V, H) * H - V);
        const float NdotL = max(dot(N, L), 0.0f);
        if (NdotL > 0.0f) {
            const float NdotH = max(dot(N, H), 0.0f);
            const float HdotV = max(dot(H, V), 0.0f);
            const float D = DistributionGGX(NdotH, pc.Roughness);
            const float pdf = max(D * NdotH / max(4.0f * HdotV, 1e-4f), 1e-4f);
            // GGX importance-sampled LOD from the sample PDF:
            // lod = 0.5 * log2(6 * width^2 / (sampleCount * pdf))
            const float mip_level = pc.Roughness == 0.0f ? 0.0f : clamp(
                0.5f * log2(6.0f * src_size * src_size / (float(SAMPLE_COUNT) * max(pdf, 1e-4f))),
                0.0f, max_src_lod
            );
            prefiltered += env_map.sample(env_sampler, L, level(mip_level)).rgb * NdotL;
            total_weight += NdotL;
        }
    }
    prefiltered = total_weight > 0.0f ? prefiltered / total_weight : float3(0.0f);
    prefilter_face.write(float4(prefiltered, 1.0f), uint2(px), face);
}

#endif
