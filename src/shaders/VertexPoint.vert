#version 450

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const DrawData draw = GetDrawData();
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint state = 0u;
    if (draw.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[draw.ElementStateSlot].States[draw.ElementStateOffset + idx];
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    Color = is_active ? vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0) :
        is_selected  ? vec4(ViewportTheme.Colors.VertexSelected, 1.0) :
                       vec4(ViewportTheme.Colors.Vertex, 1.0);
    gl_Position = SceneViewUBO.Proj * SceneViewUBO.View * world.M * vec4(vert.Position, 1.0);
    // Only show selected/active vertices in excite mode
    gl_PointSize = SceneViewUBO.InteractionMode == InteractionModeExcite && !is_selected && !is_active ? 0.0 : 6.0;
}
