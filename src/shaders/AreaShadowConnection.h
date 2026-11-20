#pragma once
#include "gpu/AreaShadowBounds.h"

// An axis-aligned box swept along the receiver-to-emitter segment contains every
// sampled connection. The slab and cross-product separating axes also work for
// parallel segments and flat boxes, without reciprocal directions.
inline bool AreaConnectionClear(float3 position, float3 to_light, float3 emitter_extent, const thread AreaShadowBounds &bounds) {
    if (!bounds.Valid) return false;
    const float3 lo = bounds.Box.Min, hi = bounds.Box.Max;
    const float3 d = 0.5f * to_light;
    const float3 m = (position - 0.5f * lo - 0.5f * hi) + d;
    // RayOrigin moves at most 256 ULPs or 1/65536 unit for a unit normal.
    // Also bound rounding in forming the sample, normalizing it and testing its segment.
    const float3 pad = 512.0f * FLT_EPSILON * (1.0f + abs(position) + abs(to_light) + emitter_extent + abs(lo) + abs(hi));
    const float3 e = (0.5f * hi - 0.5f * lo) + emitter_extent + pad;
    const float3 ad = abs(d);
    const float3 separating = abs(cross(m, d));
    const float3 projected = e.yzx * ad.zxy + e.zxy * ad.yzx;
    return any(abs(m) > e + ad) || any(separating > projected);
}
