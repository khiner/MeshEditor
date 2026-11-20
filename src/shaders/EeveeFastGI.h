/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Port of Blender 222853b0d7e eevee_fast_gi.bsl.hh and eevee_ray_types_lib.bsl.hh.
 */
#pragma once
#include <metal_stdlib>
using namespace metal;
namespace eevee_fast_gi {
// Blender eevee_bxdf_microfacet::bxdf_ggx_dominant_direction_reflection.
inline float3 dominant_reflection(float3 n, float3 v, float roughness) {
    const float m = roughness * roughness, smoothness = 1.f - m;
    return normalize(mix(n, reflect(-v, n), smoothness * (sqrt(smoothness) + m)));
}
inline uint angles_to_bitmask(float2 theta) {
    const float2 ratio = saturate(theta / M_PI_F + .5f);
    const uint a = uint(floor(32.f * ratio.x));
    const uint b = uint(ceil(32.f * (ratio.y - ratio.x)));
    // Explicitly handle the empty endpoint to avoid a shift by 32 in Metal.
    return a >= 32u ? 0u : (((b < 32u) ? 1u << b : 0u) - 1u) << a;
}
inline float bitmask_to_visibility_uniform(uint mask) { return float(popcount(mask)) / 32.f; }
inline void projected_normal(float3 N, float3 V, float3 T, float3 B, thread float &len, thread float &angle) {
    float3 projected = N - B * dot(N, B);
    len = length(projected);
    projected *= len > 0 ? 1.f / len : 0.f;
    angle = sign(dot(projected, T)) * acos(clamp(dot(projected, V), -1.f, 1.f));
}
struct ThicknessEstimator {
    float2 Slopes;
    float Depth, Time;
    float thickness(float depth, float time, float minimum) {
        const float delta = depth - Depth, slope = delta / (time - Time);
        float result = abs(delta);
        if (any(abs(Slopes - slope) > abs(Slopes))) result = 0;
        Slopes.y = Slopes.x;
        Slopes.x = slope;
        Depth = depth;
        Time = time;
        return max(result, minimum);
    }
};
// Cartesian L1 is equivalent to Blender's real SH basis after projection/evaluation.
struct SH {
    float4 L0, Lx, Ly, Lz;
    void encode(float3 n, float4 amplitude) {
        L0 += amplitude * .282094792f;
        Lx += amplitude * (.488602512f * n.x);
        Ly += amplitude * (.488602512f * n.y);
        Lz += amplitude * (.488602512f * n.z);
    }
    void add(SH s, float w) {
        L0 += s.L0 * w;
        Lx += s.Lx * w;
        Ly += s.Ly * w;
        Lz += s.Lz * w;
    }
    void scale(float4 w) {
        L0 *= w;
        Lx *= w;
        Ly *= w;
        Lz *= w;
    }
    float4 lambert(float3 n) { return max(float4(0), .282094792f * L0 + (.488602512f * 2.f / 3.f) * (Lx * n.x + Ly * n.y + Lz * n.z)); }
};
inline SH load(texture2d_array<float, access::read> image, uint2 p) { return {image.read(p, 0), image.read(p, 1), image.read(p, 2), image.read(p, 3)}; }
inline void store(texture2d_array<float, access::write> image, uint2 p, SH sh) {
    image.write(sh.L0, p, 0);
    image.write(sh.Lx, p, 1);
    image.write(sh.Ly, p, 2);
    image.write(sh.Lz, p, 3);
}
inline float3 view_point(float2 uv, float depth, float4x4 inverse) {
    const float4 p = inverse * float4(uv * float2(2, -2) + float2(-1, 1), depth, 1);
    return p.xyz / p.w;
}
struct ScreenRay {
    float2 Origin, Direction;
    float Length;
};
inline ScreenRay screen_ray(float3 origin, float3 direction, float distance, float4x4 projection, float2 size) {
    const float4 a = projection * float4(origin, 1), b = projection * float4(origin + direction * distance, 1);
    const float2 uv = a.xy / a.w * float2(.5f, -.5f) + .5f;
    float2 delta = (b.xy / b.w - a.xy / a.w) * float2(.5f, -.5f);
    if (dot(delta, delta) < 2.5e-6f) delta = float2(0, 5e-6f);
    float extent = max(abs(delta.x * size.x), abs(delta.y * size.y));
    float2 step = delta / max(extent, 1e-8f);
    const float2 boundary = select(float2(0), float2(1), step > 0);
    float2 t = select((boundary - uv) / step, float2(INFINITY), abs(step) < 1e-20f);
    return {uv, step, max(0.f, min(extent, min(t.x, t.y)))};
}
inline float pixel_thickness(float z, float4x4 projection, float pixel_radius, float constant_thickness) {
    const float footprint = pixel_radius * (4.f * sqrt(64.f / 8.f));
    if (projection[3][3] == 0) {
        const float thick = z * footprint - constant_thickness;
        return thick / (z * (z + thick)) * projection[3][2];
    }
    return -(footprint + constant_thickness) * projection[2][2];
}
} // namespace eevee_fast_gi
