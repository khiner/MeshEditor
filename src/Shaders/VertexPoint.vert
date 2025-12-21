#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const DrawData draw = DrawDataBuffers[nonuniformEXT(pc.DrawDataSlot)].Draws[gl_InstanceIndex];
    const uint vertex_count = max(draw.VertexCountOrHeadImageSlot, 1u);
    const uint idx = min(gl_VertexIndex, vertex_count - 1);
    const Vertex vert = VertexBuffers[nonuniformEXT(draw.VertexSlot)].Vertices[idx];
    const WorldMatrix world = ModelBuffers[nonuniformEXT(draw.ModelSlot)].Models[draw.ModelIndex];

    WorldNormal = mat3(world.MInv) * vert.Normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    Color = vert.Color;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
