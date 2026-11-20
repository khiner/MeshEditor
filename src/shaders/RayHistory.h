#pragma once
#include "RayGuide.h"
#include "ScreenSpace.metal"
#include "gpu/RayTracingRecord.h"

inline float4 RayHistoryGeometry(texture2d<float,access::read> geometry, uint2 pixel, float4x4 inverse) {
    const float2 stored=geometry.read(pixel).xy;
    return float4(RayWorldPoint(float2(pixel),stored.x,float2(geometry.get_width(),geometry.get_height()),inverse),stored.y);
}
inline bool RayPreviousSurface(device const RayMotionTransform *motion, uint slot, bool apply_motion,
    thread float3 &position, thread float3 &normal) {
    if (!apply_motion) return true;
    const RayMotionTransform m = motion[slot];
    if (m.Valid != RayMotionRigid) return false;
    position = (m.CurrentToPrevious.Unpack() * float4(position, 1)).xyz;
    normal = normalize(m.NormalToPrevious.Unpack() * normal);
    return all(isfinite(position)) && all(isfinite(normal));
}
// Deformation is evaluated once per visible primary surface after ray shading.
inline bool RayPreviousPixelSurface(device const RayMotionTransform *motion, uint slot, bool apply_motion,
    device const RayDeformedSurface *surfaces, uint index, thread float3 &position, thread float3 &normal) {
    if (!apply_motion || motion[slot].Valid != RayMotionDeformed) return RayPreviousSurface(motion, slot, apply_motion, position, normal);
    const RayDeformedSurface surface = surfaces[index];
    position = surface.PositionValid.xyz;
    normal = surface.Normal.xyz;
    return surface.PositionValid.w > 0.0f && all(isfinite(position)) && all(isfinite(normal));
}
inline float RayHistoryFootprint(float4x4 projection, float3 position, float height) {
    const float4 clip = projection * float4(position, 1);
    const float scale = length(float3(projection[0].y, projection[1].y, projection[2].y));
    return max(1e-5f, 2.0f * abs(clip.w) / max(height * scale, 1e-6f));
}
inline bool RayHistorySurface(float4 stored, float2 encoded_normal, uint slot, float3 position, float3 normal, float footprint, bool virtual_hit = false) {
    if ((as_type<uint>(stored.w) & 0x7fffffffu) != slot + 1u) return false;
    const float3 previous_normal = DecodeRayNormal(encoded_normal);
    if (dot(previous_normal, normal) < 0.9f) return false;
    const float3 delta = stored.xyz - position;
    if (abs(dot(delta, previous_normal)) > max(1e-4f, 0.25f * footprint)) return false;
    return virtual_hit || dot(delta, delta) < 16.0f * footprint * footprint;
}
inline bool RayFeedbackProjection(float4x4 projection, float3 position, int2 extent, thread int2 &base, thread float2 &fraction) {
    const float4 clip = projection * float4(position, 1);
    if (clip.w <= 0.0f || clip.z < 0.0f || clip.z > clip.w) return false;
    const float2 uv = ndc_to_uv(clip.xy / clip.w);
    if (any(uv < 0.0f) || any(uv >= 1.0f)) return false;
    const float2 pixel = uv * float2(extent) - 0.5f;
    base = int2(floor(pixel));
    fraction = fract(pixel);
    return true;
}
// EEVEE's previous combined-radiance feedback with validation of the Metal geometry hit.
inline bool RayFeedback(texture2d<float, access::read> radiance, texture2d<float, access::read> geometry,
    texture2d<float, access::read> normals, texture2d<uint, access::read> materials,
    device const RayMotionTransform *motion, float4x4 projection, float4x4 inverse, bool apply_motion,
    float3 position, float3 normal, uint slot, uint material, thread float3 &light) {
    if (!RayPreviousSurface(motion, slot, apply_motion, position, normal)) return false;
    const int2 extent(radiance.get_width(), radiance.get_height());
    int2 base;
    float2 f;
    if (!RayFeedbackProjection(projection, position, extent, base, f)) return false;
    const float footprint = RayHistoryFootprint(projection, position, float(extent.y));
    float3 sum(0); float weight = 0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        const int2 p = base + int2(x,y);
        if (any(p < 0) || any(p >= extent)) continue;
        if (as_type<uint>(geometry.read(uint2(p)).y) & 0x80000000u) continue;
        if (materials.read(uint2(p)).r != material + 1u || !RayHistorySurface(RayHistoryGeometry(geometry,uint2(p),inverse), normals.read(uint2(p)).xy, slot, position, normal, footprint)) continue;
        const float2 w = select(1.0f-f, f, bool2(x,y));
        sum += radiance.read(uint2(p)).rgb * w.x * w.y;
        weight += w.x * w.y;
    }
    if (weight < 0.5f || !all(isfinite(sum))) return false;
    light = sum / weight;
    return true;
}

// Current-frame opaque lighting can be validated before loading vertex attributes.
inline bool RayOpaqueFeedback(texture2d<float, access::read> radiance, texture2d<float, access::read> geometry,
    texture2d<float, access::read> normals, texture2d<uint, access::read> identity,
    float4x4 projection, float4x4 inverse, float3 outgoing, float3 position, float3 direction,
    uint slot, uint triangle, uint material, thread float3 &light, thread float3 &normal) {
    const int2 extent(radiance.get_width(), radiance.get_height());
    int2 base;
    float2 f;
    if (!RayFeedbackProjection(projection, position, extent, base, f)) return false;
    const float footprint = RayHistoryFootprint(projection, position, float(extent.y));
    float3 sum(0), accepted_normal(0), minimum(INFINITY), maximum(0); float weight = 0;
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        const int2 p = base + int2(x,y);
        if (any(p < 0) || any(p >= extent)) continue;
        if (any(identity.read(uint2(p)).xy != uint2(material + 1u, triangle + 1u))) continue;
        const float2 encoded = normals.read(uint2(p)).xy;
        const float3 n = DecodeRayNormal(encoded);
        if (dot(n, -direction) * dot(n, outgoing) <= 0.0f) continue;
        if (!RayHistorySurface(RayHistoryGeometry(geometry, uint2(p), inverse), encoded, slot, position, n, footprint)) continue;
        const float2 w = select(1.0f-f, f, bool2(x,y));
        const float3 value = radiance.read(uint2(p)).rgb;
        sum += value * w.x * w.y;
        minimum = min(minimum, value);
        maximum = max(maximum, value);
        weight += w.x * w.y;
        accepted_normal = n;
    }
    if (weight < 0.5f || !all(isfinite(sum))) return false;
    // Reuse only locally smooth lighting; preserve detail with native shading across contrast edges.
    if (any(maximum - minimum > max(maximum * 0.01f, float3(1e-4f)))) return false;
    light = sum / weight;
    normal = accepted_normal;
    return true;
}
