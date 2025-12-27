#version 450
#extension GL_EXT_nonuniform_qualifier : require

#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;

void main() {
    const uint idx = IndexBuffers[nonuniformEXT(pc.IndexSlot)].Indices[gl_VertexIndex];
    const Vertex vert = VertexBuffers[nonuniformEXT(pc.VertexSlot)].Vertices[idx];
    const WorldMatrix world = ModelBuffers[nonuniformEXT(pc.ModelSlot)].Models[pc.FirstInstance + gl_InstanceIndex];

    ElementId = pc.ElementIdOffset + idx + 1;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
    gl_PointSize = 6.0;
}
