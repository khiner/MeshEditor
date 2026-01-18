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
        state = ElementStateBuffers[draw.ElementStateSlot].States[idx];
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    Color = is_active ? ViewportTheme.Colors.ElementActive :
        is_selected  ? ViewportTheme.Colors.ElementSelected :
                       ViewportTheme.Colors.Vertex;
    gl_Position = SceneView.Proj * SceneView.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
