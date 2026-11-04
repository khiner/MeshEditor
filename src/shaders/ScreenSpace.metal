#pragma once
#include <metal_stdlib>
using namespace metal;

// Converts positive-Y-up clip coordinates to top-down framebuffer coordinates.
inline float2 ndc_to_uv(float2 ndc) { return ndc * float2(0.5f, -0.5f) + 0.5f; }

// The parameter along the edge from `from` to `to` where its screen projection crosses the knife stroke.
// Returns -1 when it misses or an end lies behind the camera.
inline float KnifeEdgeParameter(float4x4 to_clip, float2 extent, float3 from, float3 to, float2 knife_start, float2 knife_end) {
    const float4 ca = to_clip * float4(from, 1.f), cb = to_clip * float4(to, 1.f);
    if (ca.w <= 0.f || cb.w <= 0.f) return -1.f;
    const float2 a = float2(ca.x / ca.w + 1.f, 1.f - ca.y / ca.w) * 0.5f * extent, c = float2(cb.x / cb.w + 1.f, 1.f - cb.y / cb.w) * 0.5f * extent;
    const float2 d = c - a, k = knife_end - knife_start, w = knife_start - a;
    const float denominator = d.x * k.y - d.y * k.x;
    if (abs(denominator) < 1e-12f) return -1.f;
    const float t = (w.x * k.y - w.y * k.x) / denominator, u = (w.x * d.y - w.y * d.x) / denominator;
    if (t <= 0.f || t >= 1.f || u < 0.f || u > 1.f) return -1.f;
    return t;
}
