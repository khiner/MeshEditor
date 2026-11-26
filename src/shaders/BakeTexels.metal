#pragma once
// A triangle's place on a texel grid over a UV layout, and its points under texels.
#include "Bindless.metal"

// A triangle's corners in texel space, where texel (x, y) spans [x, x + 1] by [y, y + 1].
struct BakeTriangle {
    float2 Q[3];
};
inline BakeTriangle BakeTriangleOf(device const BindlessSet &b, SlotOffset uvs, uint3 corners, uint width, uint height) {
    BakeTriangle t;
    for (uint c = 0u; c < 3u; ++c) t.Q[c] = float2(BindlessBuffer(packed_float2, b.CornerUvBuffer, uvs.Slot)[uvs.Offset + corners[c]]) * float2(width, height);
    return t;
}
// Twice the triangle's signed area.
inline float BakeArea(BakeTriangle t) {
    const float2 d = t.Q[1] - t.Q[0], e = t.Q[2] - t.Q[0];
    return d.x * e.y - d.y * e.x;
}
// The corners' weights at the position.
inline float3 BakeWeights(BakeTriangle t, float2 c) {
    const float2 d = t.Q[1] - t.Q[0], e = t.Q[2] - t.Q[0], f = c - t.Q[0];
    const float area = BakeArea(t);
    const float v = (f.x * e.y - f.y * e.x) / area, w = (d.x * f.y - d.y * f.x) / area;
    return float3(1.0f - v - w, v, w);
}
// Whether the triangle covers the position, which lies inside it or on its boundary.
// Each edge's side of the position comes from the edge's ends in one order, so one of two triangles sharing an edge covers every position along it.
inline bool BakeCovers(BakeTriangle t, float2 c) {
    const float area = BakeArea(t);
    for (uint k = 0u; k < 3u; ++k) {
        const float2 a = t.Q[k], b = t.Q[(k + 1u) % 3u];
        const bool reversed = b.x < a.x || (b.x == a.x && b.y < a.y);
        const float2 p = reversed ? b : a, q = reversed ? a : b;
        const float along = (q.x - p.x) * (c.y - p.y) - (q.y - p.y) * (c.x - p.x), side = reversed ? -along : along;
        if (side != 0.0f && (side < 0.0f) != (area < 0.0f)) return false;
    }
    return area != 0.0f;
}
// The change of the corners' weights over one texel along x, and along y.
inline float3 BakeWeightsX(BakeTriangle t) {
    const float2 d = t.Q[1] - t.Q[0], e = t.Q[2] - t.Q[0];
    const float area = BakeArea(t), v = e.y / area, w = -d.y / area;
    return float3(-v - w, v, w);
}
inline float3 BakeWeightsY(BakeTriangle t) {
    const float2 d = t.Q[1] - t.Q[0], e = t.Q[2] - t.Q[0];
    const float area = BakeArea(t), v = -e.x / area, w = d.x / area;
    return float3(-v - w, v, w);
}
inline float2 BakePosition(BakeTriangle t, float3 w) { return w.x * t.Q[0] + w.y * t.Q[1] + w.z * t.Q[2]; }
// The weights of the triangle's point nearest the position: the position's own inside the triangle, and else the nearest point of its edges.
inline float3 BakeNearest(BakeTriangle t, float2 c) {
    const float3 inside = BakeWeights(t, c);
    if (all(inside >= 0.0f)) return inside;
    float nearest = INFINITY;
    float3 weights = inside;
    for (uint k = 0u; k < 3u; ++k) {
        const uint next = (k + 1u) % 3u;
        const float2 a = t.Q[k], ab = t.Q[next] - a;
        const float s = clamp(dot(c - a, ab) / max(dot(ab, ab), FLT_MIN), 0.0f, 1.0f);
        const float d = distance_squared(a + s * ab, c);
        if (d >= nearest) continue;
        nearest = d;
        weights = float3(0.0f);
        weights[k] = 1.0f - s;
        weights[next] = s;
    }
    return weights;
}
