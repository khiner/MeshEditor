#version 450

#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ElementId = draw.ElementIdOffset + gl_VertexIndex / 2 + 1;

    gl_Position = SceneViewUBO.ViewProj * vec4(trs_transform_point(world, vert.Position), 1.0);
}
