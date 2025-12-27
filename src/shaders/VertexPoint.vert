#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const uint vertex_count = max(pc.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Vertex vert = VertexBuffers[pc.VertexSlot].Vertices[idx];
    const WorldMatrix world = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex];

    uint state = 0u;
    if (pc.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[pc.ElementStateSlot].States[idx];
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    const bool is_highlighted = (state & STATE_HIGHLIGHTED) != 0u;
    const bool highlight_only = (pc.PointFlags & 1u) != 0u;
    if (highlight_only && !is_highlighted) {
        WorldNormal = vec3(0);
        WorldPosition = vec3(0);
        Color = vec4(0);
        gl_Position = vec4(2, 2, 2, 1); // Off-screen
        return;
    }

    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    Color = is_active ? Scene.ActiveColor :
        is_selected  ? Scene.SelectedColor :
        is_highlighted ? Scene.HighlightedColor :
                         Scene.VertexUnselectedColor;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
