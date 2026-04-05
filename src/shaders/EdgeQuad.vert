#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

// Signed distance from edge center in pixels (noperspective for correct screen-space interpolation).
layout(location = 0) noperspective out float EdgeCoord;
layout(location = 1) out vec4 Color;

void main() {
    const DrawData draw = GetDrawData();
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    // 6 vertices per edge quad (2 triangles).
    const uint edge_id = uint(gl_VertexIndex) / 6u;
    const uint corner_id = uint(gl_VertexIndex) % 6u;

    // Map 6 vertices to 4 unique corners: triangle 0 = {0,1,2}, triangle 1 = {1,3,2}.
    const uint corner_lut[6] = uint[6](0u, 1u, 2u, 1u, 3u, 2u);
    const uint corner = corner_lut[corner_id];

    // Corner layout:
    //   0 = endpoint0 +perp    2 = endpoint1 +perp
    //   1 = endpoint0 -perp    3 = endpoint1 -perp
    const uint endpoint = corner >> 1u;
    const float side = (corner & 1u) == 0u ? 1.0 : -1.0;

    // Read both edge endpoint vertex indices.
    const uint base_index = draw.IndexSlotOffset.Offset + edge_id * 2u;
    const uint idx0 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base_index];
    const uint idx1 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base_index + 1u];

    // Transform both endpoints to world space (morph + armature + pending transform).
    vec3 morphed0 = VertexBuffers[draw.VertexSlot].Vertices[idx0 + draw.VertexOffset].Position;
    vec3 n0 = vec3(0);
    ApplyMorphDeform(draw, morphed0, idx0, n0);
    vec3 local0 = ApplyArmatureDeform(draw, morphed0, idx0, n0);
    vec3 world0 = apply_pending_transform(draw, world, local0, idx0);

    vec3 morphed1 = VertexBuffers[draw.VertexSlot].Vertices[idx1 + draw.VertexOffset].Position;
    vec3 n1 = vec3(0);
    ApplyMorphDeform(draw, morphed1, idx1, n1);
    vec3 local1 = ApplyArmatureDeform(draw, morphed1, idx1, n1);
    vec3 world1 = apply_pending_transform(draw, world, local1, idx1);

    // Clip space.
    // NDC offset: push edges in front of faces (Blender: edge_ndc_offset_ = 1.0).
    vec4 clip0 = SceneViewUBO.ViewProj * vec4(world0, 1.0);
    vec4 clip1 = SceneViewUBO.ViewProj * vec4(world1, 1.0);
    clip0.z -= NdcOffsetFactor();
    clip1.z -= NdcOffsetFactor();

    // Near-plane clipping (z/w < -1 means behind near plane in NDC).
    vec2 pz_ndc = vec2(clip0.z / clip0.w, clip1.z / clip1.w);
    bvec2 clipped = lessThan(pz_ndc, vec2(-1.0));
    if (clipped.x && clipped.y) {
        gl_Position = vec4(uintBitsToFloat(0x7FC00000u)); // NaN discard
        return;
    }
    vec4 clip01 = clip0 - clip1;
    float ofs = abs((pz_ndc.y + 1.0) / (pz_ndc.x - pz_ndc.y));
    if (clipped.y) {
        clip1 += clip01 * ofs;
    } else if (clipped.x) {
        clip0 -= clip01 * (1.0 - ofs);
    }

    // Screen-space positions in NDC [-1,1].
    vec2 ss0 = clip0.xy / clip0.w;
    vec2 ss1 = clip1.xy / clip1.w;

    // Edge direction in pixel space, then perpendicular.
    vec2 edge_dir = (ss0 - ss1) * SceneViewUBO.ViewportSize;
    float edge_len = length(edge_dir);
    if (edge_len < 1e-6) {
        gl_Position = vec4(uintBitsToFloat(0x7FC00000u));
        return;
    }
    edge_dir /= edge_len;
    vec2 perp = vec2(-edge_dir.y, edge_dir.x);

    // Half-width: EdgeWidth is already the half-width (matches Blender's sizes.edge) + 0.5px AA fringe.
    float half_width = ViewportTheme.EdgeWidth + 0.5;

    // Offset in NDC space (pixels -> NDC).
    vec2 offset_ndc = perp * side * half_width / SceneViewUBO.ViewportSize;

    // Select clip position for this endpoint.
    vec4 pos = endpoint == 0u ? clip0 : clip1;

    // Expand in clip space (multiply by 2 because NDC range is [-1,1]).
    pos.xy += offset_ndc * 2.0 * pos.w;

    gl_Position = pos;
    EdgeCoord = side * half_width;

    // Edge selection coloring (same logic as VertexTransform.vert for edge draws).
    uint element_state = 0u;
    if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot]
            .States[draw.ElementStateSlotOffset.Offset + edge_id * 2u + endpoint]);
    }

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
