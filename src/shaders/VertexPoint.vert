#version 450

#include "Bindless.glsl"
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
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint element_state = 0u;
    if (draw.ElementState.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementState.Slot].States[draw.ElementState.Offset + idx]);
    }

    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    vec3 normal = vert.Normal;
    const vec3 local_pos = ApplyArmatureDeform(draw, vert.Position, idx, normal);
    vec3 world_pos = vec3(world.M * vec4(local_pos, 1.0));
    if (should_apply_pending_transform(draw, idx)) {
        world_pos = apply_pending_transform(local_pos, world_pos, world, draw);
    }

    WorldNormal = mat3(world.MInv) * normal;
    WorldPosition = world_pos;
    Color = is_active ? vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0) :
        is_selected  ? vec4(ViewportTheme.Colors.VertexSelected, 1.0) :
                       vec4(ViewportTheme.Colors.Vertex, 1.0);
    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
    // Only show selected/active vertices in excite mode
    gl_PointSize = SceneViewUBO.InteractionMode == InteractionModeExcite && !is_selected && !is_active ? 0.0 : 6.0;
}
