#version 450

#define BINDLESS_NO_ELEMENT_STATE 1
#define BINDLESS_CUSTOM_DRAW_PASS_PC 1
#include "SelectionElementPushConstants.glsl"
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ElementId = draw.ElementIdOffset + gl_VertexIndex / 2 + 1;

    gl_Position = SceneViewUBO.ViewProj * vec4(trs_transform_point(world, vert.Position), 1.0);
    // Slightly enlarged point fallback reduces sample-center misses on near/zero-length projected edges.
    gl_PointSize = 2.0;
}
