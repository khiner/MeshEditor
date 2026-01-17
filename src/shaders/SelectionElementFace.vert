#version 450

#define DRAW_DATA_INDEX gl_InstanceIndex
#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;
void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlot].Indices[draw.IndexOffset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    const uint face_id = ObjectIdBuffers[draw.ObjectIdSlot].Ids[gl_VertexIndex + draw.FaceIdOffset];
    ElementId = draw.ElementIdOffset + face_id;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
}
