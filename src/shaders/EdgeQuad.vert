#version 450

#include "Bindless.glsl"
#include "LineQuad.glsl"
#include "TransformUtils.glsl"

// Signed distance from edge center in pixels (noperspective for correct screen-space interpolation).
layout(location = 0) noperspective out float EdgeCoord;
layout(location = 1) out vec4 Color;
// Edge-mark band color, drawn outside the inner wire core. Zero alpha = no mark.
layout(location = 2) flat out vec4 OuterColor;

void main() {
    const DrawData draw = GetDrawData();
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    // 6 vertices per edge quad (2 triangles).
    const uint edge_id = uint(gl_VertexIndex) / 6u;
    const uint corner = line_quad_corner(uint(gl_VertexIndex));
    const uint endpoint = line_quad_endpoint(corner);
    const float side = line_quad_side(corner);

    // Read both edge endpoint vertex indices.
    const uint base_index = draw.IndexSlotOffset.Offset + edge_id * 2u;
    const uint idx0 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base_index];
    const uint idx1 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base_index + 1u];

    // Transform both endpoints to world space, from posed positions when the draw has them.
    const vec3 world0 = apply_object_pending_transform(draw, trs_transform_point(world, GetLocalPosition(draw, idx0)));
    const vec3 world1 = apply_object_pending_transform(draw, trs_transform_point(world, GetLocalPosition(draw, idx1)));

    // Clip space.
    // NDC offset: push edges in front of faces (Blender: edge_ndc_offset_ = 1.0).
    vec4 clip0 = SceneViewUBO.ViewProj * vec4(world0, 1.0);
    vec4 clip1 = SceneViewUBO.ViewProj * vec4(world1, 1.0);
    clip0.z -= NdcOffsetFactor();
    clip1.z -= NdcOffsetFactor();

    // Sharp edges draw a wider mark band around the wire core.
    const bool sharp = draw.EdgeSharpnessOffset != INVALID_OFFSET &&
        uint(ElementStateBuffers[nonuniformEXT(SceneViewUBO.EdgeSharpnessSlot)].States[draw.EdgeSharpnessOffset + edge_id]) != 0u;
    OuterColor = sharp ? vec4(ViewportTheme.Colors.EdgeSharp, 1.0) : vec4(0.0);

    // Half-width: EdgeWidth is already the half-width (matches Blender's sizes.edge) + 0.5px AA fringe.
    // Marked edges enlarge by another edge width.
    float half_width = ViewportTheme.EdgeWidth + (OuterColor.a > 0.0 ? max(ViewportTheme.EdgeWidth, 1.0) : 0.0) + 0.5;

    vec4 pos = line_quad_position(clip0, clip1, corner, half_width);
    // Marked edges draw slightly in front.
    if (OuterColor.a > 0.0) pos.z -= 5e-7 * abs(pos.w);

    gl_Position = pos;
    EdgeCoord = side * half_width;

    // Edge selection coloring (same logic as VertexTransform.vert for edge draws).
    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + edge_id * 2u + endpoint]) :
        0u;

    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionMode_Edit;
    const bool is_edit_edge = is_edit_mode && SceneViewUBO.EditElement == Element_Edge;
    const vec4 edge_color = is_edit_mode ? vec4(ViewportTheme.Colors.WireEdit, 1.0) : vec4(ViewportTheme.Colors.Wire, 1.0);
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    vec4 selected_color = edge_color;
    if (is_selected) {
        selected_color = is_edit_edge ?
            vec4(ViewportTheme.Colors.EdgeSelected, 1.0) :
            vec4(ViewportTheme.Colors.EdgeSelectedIncidental, 1.0);
    }

    vec4 final_color = is_selected ? selected_color : edge_color;
    const bool is_excited = (element_state & STATE_EXCITED) != 0u;
    if (is_excited) final_color = ViewportTheme.Colors.ElementExcited;
    else if (is_active) final_color = vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0);
    Color = final_color;
}
