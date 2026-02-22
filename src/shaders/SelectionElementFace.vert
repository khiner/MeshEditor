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
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ElementId = draw.ElementIdOffset + ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FaceIdOffset + uint(gl_VertexIndex) / 3u];

    gl_Position = SceneViewUBO.ViewProj * vec4(trs_transform_point(world, vert.Position), 1.0);
    gl_PointSize = 1.0; // Required when used with point topology for edge-on X-Ray face hits.
}
