#ifndef EDGEQUAD_MSL
#define EDGEQUAD_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "LineQuad.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "OverlayDispatch.metal"
#include "OverlayMeshPushConstants.metal"

// Edit/excite mode edges as screen-space quads with self-antialiasing.
// Each threadgroup widens up to 40 edges into 4 corners and 2 triangles apiece, one thread per corner.
constant uint EdgeQuadMeshEdges = OverlayDispatch_EdgeQuadGroupEdges;
using EdgeQuadMeshOutput = metal::mesh<EdgeQuadVaryings, void, EdgeQuadMeshEdges * 4u, EdgeQuadMeshEdges * 2u, metal::topology::triangle>;

// Matches Blender's overlay_shader_shared.hh constant values.
constant float EdgeQuadDiscRadius = 0.5641895835477563f * 1.05f;
inline float EdgeQuadSmoothWeight(float distance) {
    return smoothstep(0.5f - EdgeQuadDiscRadius, 0.5f + EdgeQuadDiscRadius, distance);
}

inline EdgeQuadVaryings EdgeQuadCorner(const thread Scene &scene, DrawData draw, uint edge_id, uint corner) {
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];
    const uint endpoint = line_quad_endpoint(corner);
    const float side = line_quad_side(corner);

    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const uint base_index = draw.IndexSlotOffset.Offset + edge_id * 2u;
    const uint idx0 = indices[base_index];
    const uint idx1 = indices[base_index + 1u];

    const float3 world0 = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx0)));
    const float3 world1 = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx1)));

    // The NDC offset pushes edges in front of faces.
    const float4x4 view_proj = scene.ViewProj();
    float4 clip0 = view_proj * float4(world0, 1.0f);
    float4 clip1 = view_proj * float4(world1, 1.0f);
    clip0.z -= NdcOffsetFactor(scene);
    clip1.z -= NdcOffsetFactor(scene);

    EdgeQuadVaryings out;
    // Sharp edges draw a wider mark band around the wire core.
    const bool sharp = draw.EdgeSharpnessOffset != INVALID_OFFSET &&
        uint(scene.ElementStates(scene.View.EdgeSharpnessSlot)[draw.EdgeSharpnessOffset + edge_id]) != 0u;
    out.OuterColor = sharp ? float4(float3(colors.EdgeSharp), 1.0f) : float4(0.0f);

    // EdgeWidth is already the half-width, plus a 0.5px antialiased fringe.
    // Marked edges enlarge by another edge width.
    const float edge_width = scene.Theme.EdgeWidth;
    const float half_width = edge_width + (out.OuterColor.a > 0.0f ? max(edge_width, 1.0f) : 0.0f) + 0.5f;

    float4 pos = line_quad_position(scene, clip0, clip1, corner, half_width);
    // Marked edges draw slightly in front.
    if (out.OuterColor.a > 0.0f) pos.z -= 5e-7f * abs(pos.w);

    out.Position = pos;
    out.EdgeCoord = side * half_width;

    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + edge_id * 2u + endpoint]) :
        0u;

    const bool is_edit_edge = scene.View.InteractionMode == InteractionMode_Edit && scene.View.EditElement == Element_Edge;
    const float4 edge_color = WireBaseColor(scene);
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    float4 selected_color = edge_color;
    if (is_selected) {
        selected_color = is_edit_edge ?
            float4(float3(colors.EdgeSelected), 1.0f) :
            float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    }

    float4 final_color = is_selected ? selected_color : edge_color;
    if (is_active) final_color = float4(float4(colors.ElementActive).rgb, 1.0f);
    out.Color = final_color;
    return out;
}

[[mesh]] void EdgeQuadMesh(
    EdgeQuadMeshOutput output,
    uint thread_index [[thread_index_in_threadgroup]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant OverlayMeshPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawDataAt(scene, pc.DrawDataIndex + threadgroup_position.y);
    if (!InstanceInFrustum(scene, draw)) {
        output.set_primitive_count(0u);
        return;
    }
    const uint first_edge = pc.FirstElement + threadgroup_position.x * EdgeQuadMeshEdges;
    const uint edge_count = min(EdgeQuadMeshEdges, pc.ElementCount - first_edge);
    output.set_primitive_count(edge_count * 2u);
    if (thread_index >= edge_count * 4u) return;

    const uint local_edge = thread_index / 4u;
    const uint corner = thread_index & 3u;
    output.set_vertex(thread_index, EdgeQuadCorner(scene, draw, first_edge + local_edge, corner));

    if (corner == 0u) {
        for (uint i = 0; i < 6u; ++i) output.set_index(local_edge * 6u + i, local_edge * 4u + LineQuadCornerLut[i]);
    }
}

inline OverlayTargets ShadeEdgeQuad(EdgeQuadVaryings in, const thread Scene &scene, bool include_outer) {
    const float edge_width = scene.Theme.EdgeWidth;
    const float dist = abs(in.EdgeCoord) - max(edge_width - 0.5f, 0.0f);
    const float mix_w = EdgeQuadSmoothWeight(dist);
    float4 color = in.Color;
    if (include_outer && in.OuterColor.a > 0.0f) {
        color = mix(in.OuterColor, color, 1.0f - mix_w);
        color.a *= 1.0f - EdgeQuadSmoothWeight(dist - max(edge_width, 1.0f));
    } else {
        color.a *= 1.0f - mix_w;
    }
    // Opt out of composite AA: edge quads handle their own.
    return OverlayTargets{color, float4(0.0f)};
}

fragment OverlayTargets EdgeQuadFragment(
    EdgeQuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return ShadeEdgeQuad(in, scene, true);
}

fragment OverlayTargets EdgeQuadSmoothFragment(
    EdgeQuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    return ShadeEdgeQuad(in, scene, false);
}

#endif
