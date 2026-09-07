#ifndef LINE_QUAD_MSL
#define LINE_QUAD_MSL

// Expands each line into a six-vertex screen-space quad.
#include "Bindless.metal"
#include "Varyings.metal"

// Triangles {0, 1, 2} and {1, 3, 2} connect positive and negative perpendicular offsets at both endpoints.
constant uint LineQuadCornerLut[6] = {0u, 1u, 2u, 1u, 3u, 2u};

inline uint line_quad_corner(uint vertex_index) { return LineQuadCornerLut[vertex_index % 6u]; }
inline uint line_quad_endpoint(uint corner) { return corner >> 1u; }
inline float line_quad_side(uint corner) { return (corner & 1u) == 0u ? 1.0f : -1.0f; }

// Metal's near plane is z = 0 in homogeneous coordinates.
inline bool ClipLineNear(thread float4 &a, thread float4 &b) {
    if (a.z < 0.0f && b.z < 0.0f) return false;
    if (a.z < 0.0f) a = mix(a, b, -a.z / (b.z - a.z));
    else if (b.z < 0.0f) b = mix(b, a, -b.z / (a.z - b.z));
    return a.w > 0.0f && b.w > 0.0f;
}

// Returns a quad corner offset `half_width` pixels from the center line.
// Clips segments against the near plane and returns NaN for fully clipped segments.
template<typename SetT>
inline float4 line_quad_position(const thread SceneT<SetT> &scene, float4 clip0, float4 clip1, uint corner, float half_width) {
    const float nan = as_type<float>(0x7FC00000u);
    if (!ClipLineNear(clip0, clip1)) return float4(nan);

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

// Analytic capsule coverage uses screen-space coordinates, including the end caps.
inline EdgeQuadVaryings StrokeQuadCorner(
    const thread Scene &scene, float4 a, float4 b, float4 color, float4 outer, float half_width, uint corner
) {
    EdgeQuadVaryings out{};
    out.Position = float4(0, 0, -1, 1);
    if (!ClipLineNear(a, b)) return out;
    const float2 extent = float2(scene.View.ViewportSize);
    const float2 delta = (b.xy / b.w - a.xy / a.w) * extent * 0.5f;
    const float len = length(delta);
    const float2 tangent = len > 1e-6f ? delta / len : float2(1, 0);
    const float2 normal = float2(-tangent.y, tangent.x);
    const bool end = line_quad_endpoint(corner) != 0u;
    const float along = end ? half_width : -half_width;
    const float across = line_quad_side(corner) * half_width;
    out.Position = end ? b : a;
    out.Position.xy += (tangent * along + normal * across) * 2.0f / extent * out.Position.w;
    out.EdgeCoord = float2((end ? len : 0.0f) + along, across);
    out.EdgeLength = len;
    out.Color = color;
    out.OuterColor = outer;
    return out;
}

template<typename Output>
inline void EmitStroke(
    thread Output output, uint index, const thread Scene &scene,
    float4 a, float4 b, float4 color, uint object_id = 0u
) {
    for (uint corner = 0u; corner < 4u; ++corner) {
        auto out = StrokeQuadCorner(scene, a, b, color, float4(0), scene.Theme.EdgeWidth + 0.5f, corner);
        out.ObjectId = object_id;
        output.set_vertex(index * 4u + corner, out);
    }
    for (uint i = 0u; i < 6u; ++i) output.set_index(index * 6u + i, index * 4u + LineQuadCornerLut[i]);
}

#endif
