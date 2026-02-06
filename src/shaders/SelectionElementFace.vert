#version 450

#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlot].Indices[draw.IndexOffset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ElementId = draw.ElementIdOffset + ObjectIdBuffers[draw.ObjectIdSlot].Ids[gl_VertexIndex + draw.FaceIdOffset];

    gl_Position = SceneViewUBO.ViewProj * (world.M * vec4(vert.Position, 1.0));
}
