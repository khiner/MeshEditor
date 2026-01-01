#version 450

#define BINDLESS_NO_ELEMENT_STATE 1
#include "Bindless.glsl"

layout(location = 0) flat out uint ElementId;

void main() {
    const uint idx = IndexBuffers[pc.IndexSlot].Indices[pc.IndexOffset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[pc.VertexSlot].Vertices[idx + pc.VertexOffset];
    const WorldMatrix world = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex];

    const uint face_id = ObjectIdBuffers[pc.ObjectIdSlot].Ids[gl_VertexIndex + pc.FaceIdOffset];
    ElementId = pc.ElementIdOffset + face_id;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
}
