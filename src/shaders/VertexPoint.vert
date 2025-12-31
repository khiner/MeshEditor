#version 450

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const uint vertex_count = max(pc.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Vertex vert = VertexBuffers[pc.VertexSlot].Vertices[idx + pc.VertexOffset];
    const WorldMatrix world = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex];

    uint state = 0u;
    if (pc.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[pc.ElementStateSlot].States[idx];
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    Color = is_active ? Scene.ActiveColor :
        is_selected  ? Scene.SelectedColor :
                       Scene.VertexUnselectedColor;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
