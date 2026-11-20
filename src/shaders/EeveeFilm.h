/* SPDX-FileCopyrightText: 2023 Blender Authors
 * SPDX-License-Identifier: GPL-2.0-or-later
 * Blender 222853b0d7e eevee_film.bsl.hh: interactive film resolve.
 */
#pragma once
#include "EeveeDenoise.h"
namespace eevee_film {
using eevee_denoise::from_ycocg;
using eevee_denoise::to_ycocg;
inline float4 color_space(float4 color) { return float4(to_ycocg(color.rgb), color.a); }
inline float luma_weight(float luma) { return 1.0f / (4.0f + luma); }
inline void catmull_weights(float2 t, thread float2 (&weights)[4]) {
    const float2 t2 = t * t, t3 = t2 * t, fct = t * .5f, fct2 = t2 * .5f, fct3 = t3 * .5f;
    weights[0] = (fct2 * 2.0f - fct3) - fct;
    weights[1] = (t3 * 2.0f - fct3) + (-t2 * 3.0f + fct2) + 1.0f;
    weights[2] = (-t3 * 2.0f + fct3) + (t2 * 3.0f - 2.0f * fct2) + fct;
    weights[3] = fct3 - fct2;
}
inline float4 sample_catmull_rom(texture2d<float> color, float2 pixel) {
    const float2 center = floor(pixel) + .5f, t = fract(pixel);
    float2 w[4];
    catmull_weights(t, w);
    const float2 w12 = w[1] + w[2];
    const float2 extent_inv = 1.0f / float2(color.get_width(), color.get_height());
    const float2 uv12 = (center + w[2] / w12) * extent_inv, uv0 = (center - 1.0f) * extent_inv, uv3 = (center + 2.0f) * extent_inv;
    const float4 cross = w12.xyyx * float4(w[0].yx, w[3].xy);
    const float middle = w12.x * w12.y;
    constexpr sampler linear(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 result = color.sample(linear, uv12, level(0)) * middle;
    result += color.sample(linear, float2(uv12.x, uv0.y), level(0)) * cross.x;
    result += color.sample(linear, float2(uv0.x, uv12.y), level(0)) * cross.y;
    result += color.sample(linear, float2(uv3.x, uv12.y), level(0)) * cross.z;
    result += color.sample(linear, float2(uv12.x, uv3.y), level(0)) * cross.w;
    return result / (middle + dot(cross, float4(1)));
}
inline void neighbor_bounds(texture2d<float, access::read> color, int2 pixel, thread float4 &low, thread float4 &high) {
    const int2 size = int2(color.get_width(), color.get_height());
    low = float4(1e16f);
    high = float4(-1e16f);
    const int2 plus[5] = {int2(0), int2(-1, 0), int2(0, -1), int2(1, 0), int2(0, 1)};
    for (uint i = 0; i < 5; ++i) {
        const float4 c = color_space(color.read(uint2(clamp(pixel + plus[i], int2(0), size - 1))));
        low = min(low, c);
        high = max(high, c);
    }
    float4 low9 = low, high9 = high;
    const int2 corners[4] = {int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1)};
    for (uint i = 0; i < 4; ++i) {
        const float4 c = color_space(color.read(uint2(clamp(pixel + corners[i], int2(0), size - 1))));
        low9 = min(low9, c);
        high9 = max(high9, c);
    }
    low = (low + low9) * .5f;
    high = (high + high9) * .5f;
}
// Five-tap variance clipping from eevee_film.bsl.hh, after Salvi (GDC 2016).
inline void variance_bounds(texture2d<float, access::read> color, int2 pixel, thread float4 &low, thread float4 &high) {
    const int2 size(color.get_width(), color.get_height());
    const int2 cross[5] = {int2(0), int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    float4 first(0), second(0);
    for (uint i = 0; i < 5; ++i) {
        const float4 c = color_space(color.read(uint2(clamp(pixel + cross[i], int2(0), size - 1))));
        first += c;
        second += c * c;
    }
    const float4 mean = 0.2f * first;
    const float4 radius = 1.25f * sqrt(abs(0.2f * second - mean * mean));
    low = mean - radius;
    high = mean + radius;
}
inline float4 amend_history(float4 low, float4 high, float4 history, float4 source) {
    const float4 direction = source - history;
    const float3 inverse = 1.0f / select(direction.rgb, float3(1e-5f), abs(direction.rgb) < 1e-5f);
    const float3 nearest = min((low.rgb - history.rgb) * inverse, (high.rgb - history.rgb) * inverse);
    history.rgb += direction.rgb * saturate(max(nearest.x, max(nearest.y, nearest.z)));
    const float t = abs(direction.a) < 1e-5f ? 0.0f : ((direction.a > 0 ? low.a : high.a) - history.a) / direction.a;
    history.a += direction.a * saturate(t);
    return history;
}
inline float history_blend(float velocity, float low, float high, float history) {
    float blend = mix(.05f, .20f, saturate(velocity * .02f));
    float distance = min(history - low, high - history);
    distance *= high != low ? 2.0f / (high - low) : 0.0f;
    return blend * saturate(distance * 4.0f + .1f);
}
inline float4 resolve(float4 source, float4 history, float4 low, float4 high, float velocity) {
    const float blend = history_blend(velocity, low.x, high.x, history.x);
    history = amend_history(low, high, history, source);
    const float wh = luma_weight(history.x) * (1.0f - blend), ws = luma_weight(source.x) * blend;
    float4 result = (history * wh + source * ws) / (wh + ws);
    result.rgb = from_ycocg(result.rgb);
    if (result.a > .995f) result.a = 1.0f;
    if (any(isnan(result))) result = float4(0, 0, 0, 1);
    return select(float4(0), result, result > 0.0f);
}
} // namespace eevee_film
