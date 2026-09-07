#ifndef WIRERASTER_MSL
#define WIRERASTER_MSL

// Each byte holds one coverage class; integer maxima make crossing wires deterministic.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "ScreenSpace.metal"
#include "WireCoverage.metal"
#include "MeshletEditGeometry.metal"
#include "WireRasterPushConstants.metal"
#include "EditSelection.metal"

// Four 8-bit coverage maxima share one word.
constant float WireCoverageScale = 255.0f;
constant float WireDiscRadius = 0.5641895835477563f * 1.05f;

inline uint WireClassOf(const thread Scene &scene, DrawData draw, uint edit_selection_color, uint edge, uint vertex_id) {
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        const uint instance_state = scene.InstanceState(draw);
        if ((instance_state & STATE_SELECTED) == 0u) return WireCoverage_Base;
        return (instance_state & STATE_ACTIVE) != 0u ? WireCoverage_Active : WireCoverage_Selected;
    }
    if (edit_selection_color == 0u || draw.Selection.Summary.Slot == INVALID_SLOT) return WireCoverage_Base;

    const uint element_state = EditEdgeEndpointState(scene, draw, edge, vertex_id);
    if ((element_state & STATE_ACTIVE) != 0u) return WireCoverage_Active;
    if ((element_state & STATE_SELECTED) == 0u) return WireCoverage_Base;
    return scene.View.InteractionMode == InteractionMode_Edit && scene.View.EditElement == Element_Edge ?
        WireCoverage_Selected :
        WireCoverage_Incidental;
}

// Union coverage within each class; higher-priority classes composite afterward.
inline void WireAccumulate(device atomic_uint *words, uint2 extent, int2 pixel, uint wire_class, float coverage) {
    if (pixel.x < 0 || pixel.y < 0 || uint(pixel.x) >= extent.x || uint(pixel.y) >= extent.y) return;
    device atomic_uint *word = &words[uint(pixel.y) * extent.x + uint(pixel.x)];
    const uint shift = wire_class * 8u;
    const uint value = uint(coverage * WireCoverageScale + 0.5f);
    uint previous = atomic_load_explicit(word, memory_order_relaxed);
    while (((previous >> shift) & 255u) < value) {
        const uint next = (previous & ~(255u << shift)) | (value << shift);
        if (atomic_compare_exchange_weak_explicit(word, &previous, next, memory_order_relaxed, memory_order_relaxed)) break;
    }
}

// Clip to Metal's six clip planes before dividing by w. The width guard retains offscreen strokes.
inline bool WireClip(thread float4 &a, thread float4 &b, float2 guard) {
    const float4 p = a, q = b;
    const float pa[6] = {p.z, p.w - p.z, p.w * guard.x + p.x, p.w * guard.x - p.x, p.w * guard.y + p.y, p.w * guard.y - p.y};
    const float pb[6] = {q.z, q.w - q.z, q.w * guard.x + q.x, q.w * guard.x - q.x, q.w * guard.y + q.y, q.w * guard.y - q.y};
    float lo = 0.0f, hi = 1.0f;
    for (uint i = 0u; i < 6u; ++i) {
        if (pa[i] < 0.0f && pb[i] < 0.0f) return false;
        if (pa[i] < 0.0f) lo = max(lo, pa[i] / (pa[i] - pb[i]));
        if (pb[i] < 0.0f) hi = min(hi, pa[i] / (pa[i] - pb[i]));
    }
    a = mix(p, q, lo); b = mix(p, q, hi);
    return lo <= hi && a.w > 0.0f && b.w > 0.0f;
}

kernel void WireRasterKernel(
    texture2d<float, access::read> visibility_depth [[texture(0)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant WireRasterPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const MeshletWork work = ResolveMeshletWork(bindless, pc.Meshlet, threadgroup_position.x);
    if (!work.Valid) return;
    const uint topology = MeshletPrimitiveTopology(work.Meshlet);
    MeshletEditEdgeGeometry geometry;
    if (topology == MeshPrimitiveTopology_Triangle) {
        const uint local_triangle = thread_index / 3u;
        const uint edge_corner = thread_index % 3u;
        if (local_triangle >= work.Meshlet.TriangleCount) return;
        const uint packed_edge = MeshletPackedEditEdge(
            bindless, pc.Meshlet, work, local_triangle, edge_corner
        );
        if (packed_edge == INVALID_OFFSET) return;
        geometry = ResolveMeshletEditEdge(
            scene, work, bindless, pc.Meshlet, local_triangle, edge_corner, packed_edge
        );
    } else if (topology == MeshPrimitiveTopology_Line) {
        if (thread_index >= work.Meshlet.TriangleCount) return;
        geometry = ResolveMeshletLineEdge(scene, work, bindless, pc.Meshlet, thread_index);
    } else {
        return;
    }

    float4 clip0 = geometry.Clip0;
    float4 clip1 = geometry.Clip1;
    // Match the hardware wire path's face depth bias.
    clip0.z -= NdcOffsetFactor(scene);
    clip1.z -= NdcOffsetFactor(scene);
    const float half_width = max(theme.EdgeWidth, 1.0f) * 0.5f;
    const float reach = half_width + WireDiscRadius;
    if (!WireClip(clip0, clip1, 1.0f + 2.0f * reach / float2(scene.View.ViewportSize))) return;

    const float2 viewport = float2(scene.View.ViewportSize);
    const uint2 extent = uint2(scene.View.ViewportSize);
    device atomic_uint *coverage_words = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.CoverageSlot);
    const float2 p0 = ndc_to_uv(clip0.xy / clip0.w) * viewport;
    const float2 p1 = ndc_to_uv(clip1.xy / clip1.w) * viewport;

    // Select the coverage class from the nearer endpoint's halfedge state.
    const uint edit_selection_color = topology == MeshPrimitiveTopology_Line ||
        scene.View.InteractionMode == InteractionMode_Edit ? 1u : 0u;
    const uint class0 = WireClassOf(
        scene, work.Draw, edit_selection_color, geometry.Edge, geometry.Vertex0
    );
    const uint class1 = WireClassOf(
        scene, work.Draw, edit_selection_color, geometry.Edge, geometry.Vertex1
    );

    const float2 delta = p1 - p0;
    const float length_px = length(delta);
    const float2 direction = length_px > 0.0f ? delta / length_px : float2(1.0f, 0.0f);

    // Step along the major axis and cover line width along the minor axis.
    const bool x_major = abs(delta.x) >= abs(delta.y);
    const float major0 = x_major ? p0.x : p0.y, major1 = x_major ? p1.x : p1.y;
    const int begin = max(0, int(floor(min(major0, major1) - reach)));
    const int end = min(int(x_major ? extent.x : extent.y) - 1, int(ceil(max(major0, major1) + reach)));
    const int spread = int(ceil(reach));
    for (int step = begin; step <= end; ++step) {
        const float t = major0 != major1 ? saturate((float(step) + 0.5f - major0) / (major1 - major0)) : 0.0f;
        const float2 at = mix(p0, p1, t);
        const int2 center = x_major ? int2(step, int(floor(at.y))) : int2(int(floor(at.x)), step);
        for (int offset = -spread; offset <= spread; ++offset) {
            const int2 pixel = x_major ? int2(center.x, center.y + offset) : int2(center.x + offset, center.y);
            const float2 sample_point = float2(pixel) + 0.5f;
            // Use segment distance to limit endpoint coverage.
            const float along = clamp(dot(sample_point - p0, direction), 0.0f, length_px);
            const float2 closest = p0 + direction * along;
            const float distance = length(sample_point - closest);
            const float coverage = smoothstep(half_width + WireDiscRadius, half_width - WireDiscRadius, distance);
            if (coverage <= 0.0f) continue;
            const float u = length_px > 0.0f ? along / length_px : 0.0f;
            if (pc.TestDepth != 0u) {
                if (any(pixel < 0) || any(uint2(pixel) >= extent)) continue;
                const float depth = mix(clip0.z / clip0.w, clip1.z / clip1.w, u);
                if (depth > visibility_depth.read(uint2(pixel)).r) continue;
            }
            WireAccumulate(coverage_words, extent, pixel, u < 0.5f ? class0 : class1, coverage);
        }
    }
}

#endif
