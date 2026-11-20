#pragma once
#include <metal_stdlib>
#include "gpu/RayReconstructPushConstants.h"
using namespace metal;
// z: roughness + 2 for glass, + 4 for deferred opaque. Negative means inactive.
inline bool RayGuideGlass(float4 guide) { return guide.z >= 2.0f && guide.z < 4.0f; }
inline bool RayGuideDeferred(float4 guide) { return guide.z >= 2.0f; }
inline float RayGuideRoughness(float4 guide) { return guide.z >= 4.0f ? guide.z - 4.0f : RayGuideGlass(guide) ? guide.z - 2.0f : guide.z; }
inline bool RayGuideSmoothGlass(float4 guide) { return RayGuideGlass(guide) && RayGuideRoughness(guide) <= 0.02f; }
inline float RayReflectionWeight(float roughness, float maximum) {
    return maximum <= 0.0f ? 0.0f : maximum >= 1.0f ? 1.0f :
                                                      1.0f - smoothstep(max(0.0f, maximum - 0.1f), maximum, roughness);
}

inline bool RayGuideActive(float4 guide, bool fast_gi, bool reflections, float maximum) {
    return guide.z >= 0.f && (!fast_gi || guide.z < 4.f || (reflections && RayReflectionWeight(guide.z - 4.f, maximum) > 0.f));
}
inline bool RayLobeActive(float4 guide, constant RayReconstructPushConstants &pc) {
    return RayGuideActive(guide, pc.FastGI, pc.Reflections, pc.MaxRoughness) &&
           (pc.Lobe == 0u || (RayGuideDeferred(guide) && guide.w >= 0.0f && (pc.Lobe < 2u || RayGuideGlass(guide))));
}
inline bool RayLobeEmpty(float4 guide, uint lobe, texture2d<float, access::read> weights, uint2 pixel) {
    return RayGuideDeferred(guide) && lobe < 2u && all(weights.read(pixel).rgb == float3(0));
}
inline float2 RaySampleOffset(uint frame) { return float2(frame) * float2(0.754877666f, 0.569840296f); }
inline uint2 RayGridOffset(uint frame, uint step) { return uint2(frame % step, (frame / step) % step); }
inline uint2 RayGridCenter(uint2 cell, uint frame, uint step, uint2 extent) {
    return min(cell * step + RayGridOffset(frame, step), extent - 1u);
}

inline float2 EncodeRayNormal(float3 n) {
    n /= abs(n.x) + abs(n.y) + abs(n.z);
    return n.z >= 0.0f ? n.xy : (1.0f - abs(n.yx)) * select(float2(-1), float2(1), n.xy >= 0.0f);
}
inline float3 DecodeRayNormal(float2 p) {
    float3 n = float3(p, 1.0f - abs(p.x) - abs(p.y));
    const float t = max(-n.z, 0.0f);
    n.xy += select(float2(t), float2(-t), n.xy >= 0.0f);
    return normalize(n);
}

inline float3 RayWorldPoint(float2 pixel, float z, float2 extent, float4x4 inverse) {
    const float2 ndc = (pixel + 0.5f) / extent * float2(2, -2) + float2(-1, 1);
    const float4 p = inverse * float4(ndc, z, 1);
    return p.xyz / p.w;
}
