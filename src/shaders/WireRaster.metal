#ifndef WIRERASTER_MSL
#define WIRERASTER_MSL

// Software line rasterization for the wireframe overlay. Each thread walks one edge and accumulates
// its coverage into a screen-sized buffer, so cost follows covered pixels rather than primitives.
// Coverage accumulates as integers, so a pixel's total does not depend on the order edges land.
// Hidden lines are rejected once per pixel: the resolve reports the nearest wire's depth and the
// overlay pass depth-tests it against the scene.
#include "Bindless.metal"
#include "SceneUBO.metal"
#include "TransformUtils.metal"
#include "ScreenSpace.metal"
#include "WireCoverage.metal"
#include "WireDrawRecord.metal"
#include "WireRasterPushConstants.metal"

// Fixed-point coverage per line, summed per class. A 32-bit counter holds any realistic line count.
constant float WireCoverageScale = 255.0f;
// How far a line's antialiased edge reaches, matching the composite's line smoothing.
constant float WireDiscRadius = 0.5641895835477563f * 1.05f;

// The class a halfedge's color comes from. The resolve turns these back into theme colors.
inline uint WireClassOf(const thread Scene &scene, DrawData draw, uint element_state_color, uint vertex_index) {
    if (scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        const uint instance_state = scene.InstanceState(draw);
        if ((instance_state & STATE_SELECTED) == 0u) return WireCoverage_Base;
        return (instance_state & STATE_ACTIVE) != 0u ? WireCoverage_Active : WireCoverage_Selected;
    }
    if (element_state_color == 0u || draw.ElementStateSlotOffset.Slot == INVALID_SLOT) return WireCoverage_Base;

    const uint element_state = uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + vertex_index]);
    if ((element_state & STATE_ACTIVE) != 0u) return WireCoverage_Active;
    if ((element_state & STATE_SELECTED) == 0u) return WireCoverage_Base;
    return scene.View.InteractionMode == InteractionMode_Edit && scene.View.EditElement == Element_Edge ?
        WireCoverage_Selected :
        WireCoverage_Incidental;
}

// The depth word holds the IEEE bits complemented, so a maximum keeps the nearest line and a
// zero-filled buffer starts out empty.
inline void WireAccumulate(
    device atomic_uint *words, uint2 extent, int2 pixel, uint wire_class, float coverage, float depth
) {
    if (pixel.x < 0 || pixel.y < 0 || uint(pixel.x) >= extent.x || uint(pixel.y) >= extent.y) return;
    const uint base = (uint(pixel.y) * extent.x + uint(pixel.x)) * WireCoverage_WordsPerPixel;
    atomic_fetch_add_explicit(&words[base + wire_class], uint(coverage * WireCoverageScale + 0.5f), memory_order_relaxed);
    atomic_fetch_max_explicit(&words[base + WireCoverage_DepthWord], ~as_type<uint>(depth), memory_order_relaxed);
}

// Clip a segment against the near plane so a line crossing behind the eye still draws its visible part.
constant float WireNearEpsilon = 1e-5f;

inline bool WireClipNear(thread float4 &a, thread float4 &b) {
    const bool a_in = a.w > WireNearEpsilon, b_in = b.w > WireNearEpsilon;
    if (!a_in && !b_in) return false;
    if (a_in && b_in) return true;
    const float t = (WireNearEpsilon - a.w) / (b.w - a.w);
    const float4 crossing = a + (b - a) * t;
    if (a_in) b = crossing;
    else a = crossing;
    return true;
}

// The record a global edge index falls in: the last one whose flattened range has begun.
inline uint WireRecordAt(device const WireDrawRecord *records, uint count, uint global_edge) {
    uint lo = 0u, hi = count - 1u;
    while (lo < hi) {
        const uint mid = (lo + hi + 1u) / 2u;
        if (records[mid].FirstEdge <= global_edge) lo = mid;
        else hi = mid - 1u;
    }
    return lo;
}

kernel void WireRasterKernel(
    uint3 thread_position [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant WireRasterPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const uint global_edge = thread_position.x;
    if (global_edge >= pc.EdgeTotal) return;

    const Scene scene{bindless, view, theme, workspace};
    device const WireDrawRecord *records = BindlessBuffer(WireDrawRecord, bindless.Buffer, pc.RecordSlot);
    const WireDrawRecord record = records[WireRecordAt(records, pc.RecordCount, global_edge)];
    // Every instance of a draw repeats its edge list, so the remainder splits into instance and edge.
    const uint local = global_edge - record.FirstEdge;
    const uint instance = local / record.EdgeCount;
    const uint edge = local - instance * record.EdgeCount;

    const DrawData draw = GetDrawDataAt(scene, record.DrawDataIndex + instance);
    if (!InstanceInFrustum(scene, draw)) return;

    const uint index_position = edge * 2u;
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    const uint i0 = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + index_position];
    const uint i1 = scene.Indices(draw.IndexSlotOffset.Slot)[draw.IndexSlotOffset.Offset + index_position + 1u];
    float4 clip0 = MeshletPosition(scene, draw, world, i0);
    float4 clip1 = MeshletPosition(scene, draw, world, i1);
    // Push lines in front of faces, matching the hardware wire emission.
    clip0.z -= NdcOffsetFactor(scene);
    clip1.z -= NdcOffsetFactor(scene);
    if (!WireClipNear(clip0, clip1)) return;

    const float2 viewport = float2(scene.View.ViewportSize);
    const uint2 extent = uint2(scene.View.ViewportSize);
    device atomic_uint *coverage_words = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.CoverageSlot);
    const float2 p0 = ndc_to_uv(clip0.xy / clip0.w) * viewport;
    const float2 p1 = ndc_to_uv(clip1.xy / clip1.w) * viewport;
    const float z0 = clip0.z / clip0.w, z1 = clip1.z / clip1.w;

    // Each halfedge carries its own state, so the class comes from the endpoint the pixel is nearer.
    const uint class0 = WireClassOf(scene, draw, record.ElementStateColor, index_position);
    const uint class1 = WireClassOf(scene, draw, record.ElementStateColor, index_position + 1u);

    const float half_width = max(theme.EdgeWidth, 1.0f) * 0.5f;
    const float reach = half_width + WireDiscRadius;
    const float2 delta = p1 - p0;
    const float length_px = length(delta);
    const float2 direction = length_px > 0.0f ? delta / length_px : float2(1.0f, 0.0f);

    // Walk the major axis one pixel at a time, covering the line's width along the minor axis.
    // A sub-pixel line's endpoints land in one pixel, so the walk splats each center once.
    const bool x_major = abs(delta.x) >= abs(delta.y);
    const int steps = int(min(max(abs(x_major ? delta.x : delta.y), 1.0f), 4096.0f));
    const int spread = int(ceil(reach));
    int2 previous = int2(INT_MIN);
    for (int step = 0; step <= steps; ++step) {
        const float t = float(step) / float(steps);
        const float2 at = p0 + delta * t;
        const int2 center = int2(floor(at));
        if (all(center == previous)) continue;
        previous = center;
        for (int offset = -spread; offset <= spread; ++offset) {
            const int2 pixel = x_major ? int2(center.x, center.y + offset) : int2(center.x + offset, center.y);
            const float2 sample_point = float2(pixel) + 0.5f;
            // Distance to the segment, so an endpoint pixel does not leak coverage past the line's end.
            const float along = clamp(dot(sample_point - p0, direction), 0.0f, length_px);
            const float2 closest = p0 + direction * along;
            const float distance = length(sample_point - closest);
            const float coverage = smoothstep(half_width + WireDiscRadius, half_width - WireDiscRadius, distance);
            if (coverage <= 0.0f) continue;
            const float u = length_px > 0.0f ? along / length_px : 0.0f;
            const float depth = mix(z0, z1, u);
            WireAccumulate(coverage_words, extent, pixel, u < 0.5f ? class0 : class1, coverage, depth);
        }
        if (length_px <= 0.0f) break;
    }
}

#endif
