#version 450
#extension GL_EXT_nonuniform_qualifier : require
#include "Bindless.glsl"

layout(location = 0) flat out uint ObjectId;
layout(location = 1) flat out uint HeadImageSlot;
layout(location = 2) flat out uint SelectionNodesSlot;
layout(location = 3) flat out uint SelectionCounterSlot;

void main() {
    const DrawData draw = DrawDataBuffers[nonuniformEXT(pc.DrawDataSlot)].Draws[gl_InstanceIndex];
    const uint idx = IndexBuffers[nonuniformEXT(draw.IndexSlot)].Indices[gl_VertexIndex];
    const vec3 position = VertexBuffers[nonuniformEXT(draw.VertexSlot)].Vertices[idx].Position;
    const mat4 model = ModelBuffers[nonuniformEXT(draw.ModelSlot)].Models[draw.ModelIndex].M;
    ObjectId = draw.ObjectId;
    HeadImageSlot = draw.VertexCountOrHeadImageSlot;
    SelectionNodesSlot = draw.SelectionNodesSlot;
    SelectionCounterSlot = draw.SelectionCounterSlot;
    gl_Position = Scene.Proj * Scene.View * model * vec4(position, 1.0);
}
