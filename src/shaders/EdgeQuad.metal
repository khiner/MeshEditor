#ifndef EDGEQUAD_MSL
#define EDGEQUAD_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "LineQuad.metal"
#include "TransformUtils.metal"
#include "Varyings.metal"
#include "MainDrawPushConstants.metal"

vertex EdgeQuadVaryings EdgeQuadVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MainDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const Transform world = scene.Models(draw.ModelSlot)[draw.FirstInstance];

    // Six vertices per edge quad (two triangles).
    const uint edge_id = vertex_id / 6u;
    const uint corner = line_quad_corner(vertex_id);
    const uint endpoint = line_quad_endpoint(corner);
    const float side = line_quad_side(corner);

    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const uint base_index = draw.IndexSlotOffset.Offset + edge_id * 2u;
    const uint idx0 = indices[base_index];
    const uint idx1 = indices[base_index + 1u];

    // Both endpoints in world space, from posed positions when the draw has them.
    const float3 world0 = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx0)));
    const float3 world1 = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, idx1)));

    // Clip space, with the NDC offset pushing edges in front of faces.
    const float4x4 view_proj = scene.ViewProj();
    float4 clip0 = view_proj * float4(world0, 1.0f);
    float4 clip1 = view_proj * float4(world1, 1.0f);
    clip0.z -= NdcOffsetFactor(scene);
    clip1.z -= NdcOffsetFactor(scene);

    EdgeQuadVaryings out;
    // Sharp edges draw a wider mark band around the wire core.
    const bool sharp = draw.EdgeSharpnessOffset != INVALID_OFFSET &&
        uint(scene.ElementStates(view.EdgeSharpnessSlot)[draw.EdgeSharpnessOffset + edge_id]) != 0u;
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

    // Edge selection coloring, matching the mesh transform's edge draws.
    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + edge_id * 2u + endpoint]) :
        0u;

    const bool is_edit_mode = view.InteractionMode == InteractionMode_Edit;
    const bool is_edit_edge = is_edit_mode && view.EditElement == Element_Edge;
    const float4 edge_color = is_edit_mode ? float4(float3(colors.WireEdit), 1.0f) : float4(float3(colors.Wire), 1.0f);
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    float4 selected_color = edge_color;
    if (is_selected) {
        selected_color = is_edit_edge ?
            float4(float3(colors.EdgeSelected), 1.0f) :
            float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    }

    float4 final_color = is_selected ? selected_color : edge_color;
    const bool is_excited = (element_state & STATE_EXCITED) != 0u;
    if (is_excited) final_color = float4(colors.ElementExcited);
    else if (is_active) final_color = float4(float4(colors.ElementActive).rgb, 1.0f);
    out.Color = final_color;
    return out;
}

fragment OverlayTargets EdgeQuadFragment(
    EdgeQuadVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    // Matches Blender's overlay_shader_shared.hh constant values.
    const float DISC_RADIUS = 0.5641895835477563f * 1.05f; // M_1_SQRTPI * 1.05
    const float LINE_SMOOTH_START = 0.5f - DISC_RADIUS;
    const float LINE_SMOOTH_END = 0.5f + DISC_RADIUS;

    const float edge_width = scene.Theme.EdgeWidth;
    const float dist = abs(in.EdgeCoord) - max(edge_width - 0.5f, 0.0f);
    const float dist_outer = dist - max(edge_width, 1.0f);
    const float mix_w = smoothstep(LINE_SMOOTH_START, LINE_SMOOTH_END, dist);
    const float mix_w_outer = smoothstep(LINE_SMOOTH_START, LINE_SMOOTH_END, dist_outer);

    float4 color = mix(in.OuterColor, in.Color, 1.0f - mix_w * in.OuterColor.a);
    color.a *= 1.0f - (in.OuterColor.a > 0.0f ? mix_w_outer : mix_w);
    // Opt out of composite AA: edge quads handle their own.
    return OverlayTargets{color, float4(0.0f)};
}

#endif
