#version 450

#include "Bindless.glsl"
#include "TransformUtils.glsl"

layout(location = 0) out vec4 Color;

// Object mode has no element selection, so points take their object's selection color.
vec4 point_color(DrawData draw, uint element_state) {
    if (SceneViewUBO.InteractionMode == InteractionMode_Object && SceneViewUBO.ShowOverlays != 0u) {
        return ObjectSelectionColor(InstanceState(draw), vec4(ViewportTheme.Colors.Vertex, 1.0));
    }
    if ((element_state & STATE_EXCITED) != 0u) return ViewportTheme.Colors.ElementExcited;
    if ((element_state & STATE_ACTIVE) != 0u) return vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0);
    if ((element_state & STATE_SELECTED) != 0u) return vec4(ViewportTheme.Colors.VertexSelected, 1.0);
    return vec4(ViewportTheme.Colors.Vertex, 1.0);
}

void main() {
    const DrawData draw = GetDrawData();
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT ?
        uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + idx]) :
        0u;

    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    const vec3 local_pos = GetLocalPosition(draw, idx);
    const vec3 world_pos = apply_object_pending_transform(draw, trs_transform_point(world, local_pos));

    Color = point_color(draw, element_state);
    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    gl_Position.z -= NdcOffsetFactor() * 1.5; // Push points in front of lines/faces (Blender: vertex_ndc_offset_)
    // Only show selected/active vertices in excite mode
    gl_PointSize = SceneViewUBO.InteractionMode == InteractionMode_Excite && !is_selected && !is_active ? 0.0 : PointSize;
}
