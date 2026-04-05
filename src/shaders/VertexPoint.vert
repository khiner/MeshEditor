#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const DrawData draw = GetDrawData();
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint element_state = 0u;
    if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + idx]);
    }

    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    vec3 normal = vert.Normal;
    vec3 morphed_pos = vert.Position;
    ApplyMorphDeform(draw, morphed_pos, idx, normal);
    const vec3 local_pos = ApplyArmatureDeform(draw, morphed_pos, idx, normal);
    vec3 world_pos = apply_pending_transform(draw, world, local_pos, idx);

    WorldNormal = trs_transform_normal(world, normal);
    WorldPosition = world_pos;
    const bool is_excited = (element_state & STATE_EXCITED) != 0u;
    Color = is_excited   ? ViewportTheme.Colors.ElementExcited :
        is_active        ? vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0) :
        is_selected      ? vec4(ViewportTheme.Colors.VertexSelected, 1.0) :
                           vec4(ViewportTheme.Colors.Vertex, 1.0);
    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    gl_Position.z -= NdcOffsetFactor() * 1.5; // Push points in front of lines/faces (Blender: vertex_ndc_offset_)
    // Only show selected/active vertices in excite mode
    gl_PointSize = SceneViewUBO.InteractionMode == InteractionMode_Excite && !is_selected && !is_active ? 0.0 : 8.0;
}
