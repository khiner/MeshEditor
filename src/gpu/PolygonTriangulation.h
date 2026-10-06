#pragma once

#include "gpu/Types.h"
#ifndef __METAL_VERSION__
#include "numeric/VectorMath.h"
#endif

// Shared by source creation, canonical render triangles, and the Triangulate tool.
inline float PolygonCross2(vec2 a, vec2 b, vec2 c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

// Ear-clips a projected polygon in a fixed order. The caller owns its scratch
// links and points, in host memory for source creation or GPU scratch for edits.
template<typename Points, typename Next, typename Previous, typename Emit>
inline void PolygonEarClip(Points points, Next next, Previous prev, uint32_t n, Emit emit) {
    for (uint32_t i = 0u; i < n; ++i) {
        next[i] = (i + 1u) % n;
        prev[i] = (i + n - 1u) % n;
    }
    float area = 0.f;
    for (uint32_t i = 0u; i < n; ++i) area += points[i].x * points[next[i]].y - points[next[i]].x * points[i].y;
    const float sign = area >= 0.f ? 1.f : -1.f;
    uint32_t count = 0u, remaining = n, cursor = 0u;
    while (remaining > 3u) {
        for (uint32_t attempt = 0u; attempt < remaining; ++attempt) {
            const uint32_t i = cursor, a = prev[i], b = next[i];
            const vec2 pa = points[a], pi = points[i], pb = points[b];
            const bool convex = sign * PolygonCross2(pa, pi, pb) > 0.f;
            bool ear = convex;
            for (uint32_t j = next[b]; ear && j != a; j = next[j]) {
                const vec2 q = points[j];
                const bool inside = sign * PolygonCross2(pa, pi, q) >= 0.f && sign * PolygonCross2(pi, pb, q) >= 0.f && sign * PolygonCross2(pb, pa, q) >= 0.f;
                if (inside) ear = false;
            }
            if (ear) break;
            cursor = b;
        }
        // A full unsuccessful scan returns to the cursor; clip it for a degenerate polygon.
        const uint32_t a = prev[cursor], b = next[cursor];
        emit(uvec3(a, cursor, b), count++);
        next[a] = b;
        prev[b] = a;
        --remaining;
        cursor = b;
    }
    emit(uvec3(prev[cursor], cursor, next[cursor]), count);
}

// Project relative to the first corner and scale before computing a vector-area
// normal. This avoids translation cancellation and supports either winding and
// polygons in any plane. Convex loops retain their original fan ordering.
template<typename Position, typename Points, typename Next, typename Previous, typename Emit>
inline void TriangulatePolygon(uint32_t n, Position position, Points points, Next next, Previous previous, Emit emit) {
    if (n < 3u) return;
    if (n == 3u) {
        emit(uvec3{0u, 1u, 2u}, 0u);
        return;
    }
    const vec3 origin = position(0u);
    vec3 low = origin, high = origin;
    for (uint32_t i = 1u; i < n; ++i) {
        const vec3 p = position(i);
        for (uint32_t k = 0u; k < 3u; ++k) {
            low[k] = p[k] < low[k] ? p[k] : low[k];
            high[k] = p[k] > high[k] ? p[k] : high[k];
        }
    }
    const vec3 extent = high - low;
    float scale = extent.x > extent.y ? extent.x : extent.y;
    scale = scale > extent.z ? scale : extent.z;
    if (!(scale > 0.f)) scale = 1.f;
    vec3 normal{0.f, 0.f, 0.f}, a{0.f, 0.f, 0.f};
    for (uint32_t i = 1u; i < n; ++i) {
        const vec3 b = (position(i) - origin) / scale;
        normal.x += a.y * b.z - a.z * b.y;
        normal.y += a.z * b.x - a.x * b.z;
        normal.z += a.x * b.y - a.y * b.x;
        a = b;
    }
    const auto magnitude = [](float value) { return value < 0.f ? -value : value; };
    uint32_t axis = magnitude(normal.y) > magnitude(normal.x) ? 1u : 0u;
    if (magnitude(normal.z) > magnitude(normal[axis])) axis = 2u;
    if (!(magnitude(normal[axis]) > 0.f)) {
        axis = extent.y < extent.x ? 1u : 0u;
        if (extent.z < extent[axis]) axis = 2u;
    }
    const uint32_t x = (axis + 1u) % 3u, y = (axis + 2u) % 3u;
    for (uint32_t i = 0u; i < n; ++i) {
        const vec3 p = (position(i) - origin) / scale;
        points[i] = vec2{p[x], p[y]};
    }
    float area = 0.f;
    for (uint32_t i = 0u; i < n; ++i) area += PolygonCross2(vec2{0.f, 0.f}, points[i], points[(i + 1u) % n]);
    const float sign = area >= 0.f ? 1.f : -1.f;
    bool convex = true;
    for (uint32_t i = 0u; i < n; ++i) convex = convex && sign * PolygonCross2(points[(i + n - 1u) % n], points[i], points[(i + 1u) % n]) >= 0.f;
    if (convex) {
        for (uint32_t i = 1u; i + 1u < n; ++i) emit(uvec3{0u, i, i + 1u}, i - 1u);
    } else PolygonEarClip(points, next, previous, n, emit);
}
