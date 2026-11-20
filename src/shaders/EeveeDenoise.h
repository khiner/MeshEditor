/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Metal port of eevee_ray_denoise.bsl.hh, eevee_colorspace_lib.bsl.hh and
 * eevee_filter.bsl.hh from Blender 222853b0d7e.
 */
#pragma once
#include <metal_stdlib>
using namespace metal;
namespace eevee_denoise {
inline float3 to_ycocg(float3 c) { return float3(c.r + 2.0f * c.g + c.b, 2.0f * c.r - 2.0f * c.b, -c.r + 2.0f * c.g - c.b); }
inline float3 from_ycocg(float3 c) { return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z) * 0.25f; }
inline float3 to_log(float3 c) { return log2(1.0f + max(c, 0.0f)); }
inline float3 from_log(float3 c) { return exp2(c) - 1.0f; }
inline float3 safe_color(float3 c) { return select(float3(0), clamp(c, 0.0f, 65504.0f), isfinite(c)); }
inline float gaussian_factor(float radius, float deviation) { return 1.442695041f * deviation / (radius * radius); }
inline float gaussian_weight(float factor, float squared_distance) { return exp2(-factor * squared_distance); }
inline float planar_weight(float3 n, float3 p, float3 other) {
    float d = dot(n, other - p);
    return gaussian_weight(10000.0f, d * d);
}
inline float angle_weight(float3 n, float3 other) {
    float d = dot(n, other);
    d *= d;
    d *= d;
    return saturate(d * d);
}
inline float2 hammersley(uint i, uint count) { return float2(float(i) / float(count), float(reverse_bits(i)) * 2.3283064365386963e-10f); }
inline float2 sample_disk(float2 u) {
    float a = 6.283185307179586f * (u.y - 0.5f);
    return sqrt(u.x) * float2(cos(a), sin(a));
}
inline float noise(float2 p) { return fract(52.9829189f * fract(dot(p, float2(0.06711056f, 0.00583715f)))); }
struct LocalStatistics {
    float3 mean, moment, variance, deviation, clamp_min, clamp_max;
};
inline LocalStatistics finish_statistics(float3 mean, float3 moment, float count) {
    LocalStatistics s;
    s.mean = mean / count;
    s.moment = moment / count;
    s.variance = abs(s.moment - s.mean * s.mean);
    s.deviation = max(float3(1e-4f), sqrt(s.variance));
    s.clamp_min = s.mean - s.deviation;
    s.clamp_max = s.mean + s.deviation;
    return s;
}
inline float history_weight(float3 color, LocalStatistics s) {
    return exp2(-4.0f * dot(abs(color - s.mean) / s.deviation, float3(1)));
}
inline float spatial_variance(float3 mean, float3 moment) {
    float3 v = abs(moment - mean * mean) / max(mean, float3(1e-20f));
    return max(v.x, max(v.y, v.z));
}
inline float temporal_mix(float variance, float valid_weight) {
    return valid_weight > 1e-3f ? 0.97f * mix(0.75f, 1.0f, saturate(variance * 20.0f)) : 0.0f;
}
inline float temporal_variance(float current, float previous, bool valid) {
    previous = clamp(previous, 0.0f, 100.0f);
    return sqrt(mix(current * current, previous * previous, valid ? 0.85f : 0.0f));
}
} // namespace eevee_denoise
