#version 450

#include "Bindless.glsl"
#include "ObjectExtrasTransform.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;
layout(location = 3) flat out uint FaceOverlayFlags;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexOffset.Slot].Indices[draw.IndexOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    const vec3 world_pos = ObjectExtrasWorldPos(draw, vert, world, idx);

    WorldNormal = vec3(0, 1, 0);
    WorldPosition = world_pos;
    FaceOverlayFlags = 0u;

    uint instance_state = 0u;
    if (draw.InstanceStateOffset.Slot != INVALID_SLOT) {
        instance_state = uint(InstanceStateBuffers[draw.InstanceStateOffset.Slot].States[draw.InstanceStateOffset.Offset + draw.FirstInstance]);
    }

    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionModeEdit;
    const vec4 wire_color = is_edit_mode ? vec4(ViewportTheme.Colors.WireEdit, 1.0) : vec4(ViewportTheme.Colors.Wire, 1.0);
    const bool is_selected = (instance_state & STATE_SELECTED) != 0u;
    const bool is_active = (instance_state & STATE_ACTIVE) != 0u;

    vec4 final_color = wire_color;
    if (is_selected) final_color = vec4(ViewportTheme.Colors.ObjectSelected, 1.0);
    if (is_active) final_color = vec4(ViewportTheme.Colors.ObjectActive, 1.0);

    // Ground line/diamond: fixed theme color, unaffected by selection state.
    if (GetVertexClass(draw, idx) == VCLASS_GROUNDPOINT) {
        final_color = ViewportTheme.Colors.Light;
    }
    Color = final_color;

    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
}
