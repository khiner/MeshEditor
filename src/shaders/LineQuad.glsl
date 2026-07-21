#ifndef LINE_QUAD_GLSL
#define LINE_QUAD_GLSL

// Widening of a line into a screen-space quad: 6 vertices per line, mapping to 4 unique corners.
// Requires: SceneUBO.glsl (for SceneViewUBO)

// Triangle 0 = {0,1,2}, triangle 1 = {1,3,2}, over the corners:
//   0 = endpoint0 +perp    2 = endpoint1 +perp
//   1 = endpoint0 -perp    3 = endpoint1 -perp
uint line_quad_corner(uint vertex_index) {
    const uint corner_lut[6] = uint[6](0u, 1u, 2u, 1u, 3u, 2u);
    return corner_lut[vertex_index % 6u];
}
uint line_quad_endpoint(uint corner) { return corner >> 1u; }
float line_quad_side(uint corner) { return (corner & 1u) == 0u ? 1.0 : -1.0; }

// Clip position for one corner of the quad around the segment, `half_width` pixels off its center line.
// A segment crossing the near plane draws from the crossing point, and one entirely behind it discards.
vec4 line_quad_position(vec4 clip0, vec4 clip1, uint corner, float half_width) {
    const float nan = uintBitsToFloat(0x7FC00000u); // NaN positions discard the primitive.
    // Near-plane clipping (z/w < -1 means behind the near plane in NDC).
    const vec2 pz_ndc = vec2(clip0.z / clip0.w, clip1.z / clip1.w);
    const bvec2 clipped = lessThan(pz_ndc, vec2(-1.0));
    if (clipped.x && clipped.y) return vec4(nan);

    const vec4 clip01 = clip0 - clip1;
    const float ofs = abs((pz_ndc.y + 1.0) / (pz_ndc.x - pz_ndc.y));
    if (clipped.y) clip1 += clip01 * ofs;
    else if (clipped.x) clip0 -= clip01 * (1.0 - ofs);

    // Segment direction in pixel space, then its perpendicular.
    vec2 dir = (clip0.xy / clip0.w - clip1.xy / clip1.w) * SceneViewUBO.ViewportSize;
    const float len = length(dir);
    if (len < 1e-6) return vec4(nan);
    dir /= len;
    const vec2 perp = vec2(-dir.y, dir.x);

    vec4 pos = line_quad_endpoint(corner) == 0u ? clip0 : clip1;
    // Expand in clip space, doubling the pixel offset because NDC spans [-1,1].
    pos.xy += perp * line_quad_side(corner) * half_width / SceneViewUBO.ViewportSize * 2.0 * pos.w;
    return pos;
}

#endif
