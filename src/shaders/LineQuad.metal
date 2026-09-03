#ifndef LINE_QUAD_MSL
#define LINE_QUAD_MSL

// Expands each line into a six-vertex screen-space quad.
#include "Bindless.metal"

// Triangles {0, 1, 2} and {1, 3, 2} connect positive and negative perpendicular offsets at both endpoints.
constant uint LineQuadCornerLut[6] = {0u, 1u, 2u, 1u, 3u, 2u};

inline uint line_quad_corner(uint vertex_index) { return LineQuadCornerLut[vertex_index % 6u]; }
inline uint line_quad_endpoint(uint corner) { return corner >> 1u; }
inline float line_quad_side(uint corner) { return (corner & 1u) == 0u ? 1.0f : -1.0f; }

// Returns a quad corner offset `half_width` pixels from the center line.
// Clips segments against the near plane and returns NaN for fully clipped segments.
template<typename SetT>
inline float4 line_quad_position(const thread SceneT<SetT> &scene, float4 clip0, float4 clip1, uint corner, float half_width) {
    const float nan = as_type<float>(0x7FC00000u);
    const float2 pz_ndc = float2(clip0.z / clip0.w, clip1.z / clip1.w);
    const bool2 clipped = pz_ndc < float2(-1.0f);
    if (clipped.x && clipped.y) return float4(nan);

    const float4 clip01 = clip0 - clip1;
    const float ofs = abs((pz_ndc.y + 1.0f) / (pz_ndc.x - pz_ndc.y));
    if (clipped.y) clip1 += clip01 * ofs;
    else if (clipped.x) clip0 -= clip01 * (1.0f - ofs);

    // Preserve clip-space positive Y because the corner sign absorbs perpendicular orientation.
    const float2 viewport_size = float2(scene.View.ViewportSize);
    float2 dir = (clip0.xy / clip0.w - clip1.xy / clip1.w) * viewport_size;
    const float len = length(dir);
    if (len < 1e-6f) return float4(nan);
    dir /= len;
    const float2 perp = float2(-dir.y, dir.x);

    float4 pos = line_quad_endpoint(corner) == 0u ? clip0 : clip1;
    // Double the pixel offset because NDC spans [-1, 1].
    pos.xy += perp * line_quad_side(corner) * half_width / viewport_size * 2.0f * pos.w;
    return pos;
}

#endif
