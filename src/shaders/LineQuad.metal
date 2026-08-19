#ifndef LINE_QUAD_MSL
#define LINE_QUAD_MSL

// Widening of a line into a screen-space quad: 6 vertices per line, mapping to 4 unique corners.
#include "Bindless.metal"

// Triangle 0 = {0,1,2}, triangle 1 = {1,3,2}, over the corners:
//   0 = endpoint0 +perp    2 = endpoint1 +perp
//   1 = endpoint0 -perp    3 = endpoint1 -perp
constant uint LineQuadCornerLut[6] = {0u, 1u, 2u, 1u, 3u, 2u};

inline uint line_quad_corner(uint vertex_index) { return LineQuadCornerLut[vertex_index % 6u]; }
inline uint line_quad_endpoint(uint corner) { return corner >> 1u; }
inline float line_quad_side(uint corner) { return (corner & 1u) == 0u ? 1.0f : -1.0f; }

// Clip position for one corner of the quad around the segment, `half_width` pixels off its center line.
// A segment crossing the near plane draws from the crossing point, and one entirely behind it discards.
template<typename SetT>
inline float4 line_quad_position(const thread SceneT<SetT> &scene, float4 clip0, float4 clip1, uint corner, float half_width) {
    const float nan = as_type<float>(0x7FC00000u); // NaN positions discard the primitive.
    // Near-plane clipping (z/w < -1 means behind the near plane in NDC).
    const float2 pz_ndc = float2(clip0.z / clip0.w, clip1.z / clip1.w);
    const bool2 clipped = pz_ndc < float2(-1.0f);
    if (clipped.x && clipped.y) return float4(nan);

    const float4 clip01 = clip0 - clip1;
    const float ofs = abs((pz_ndc.y + 1.0f) / (pz_ndc.x - pz_ndc.y));
    if (clipped.y) clip1 += clip01 * ofs;
    else if (clipped.x) clip0 -= clip01 * (1.0f - ofs);

    // Segment direction scaled to pixels, then its perpendicular.
    // Only the length and the perpendicular are taken, and the corner's side sign absorbs which way
    // the perpendicular turns, so the scale leaves y pointing up as clip space has it.
    const float2 viewport_size = float2(scene.View.ViewportSize);
    float2 dir = (clip0.xy / clip0.w - clip1.xy / clip1.w) * viewport_size;
    const float len = length(dir);
    if (len < 1e-6f) return float4(nan);
    dir /= len;
    const float2 perp = float2(-dir.y, dir.x);

    float4 pos = line_quad_endpoint(corner) == 0u ? clip0 : clip1;
    // Expand in clip space, doubling the pixel offset because NDC spans [-1,1].
    pos.xy += perp * line_quad_side(corner) * half_width / viewport_size * 2.0f * pos.w;
    return pos;
}

#endif
