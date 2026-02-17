#version 450

#include "Bindless.glsl"
#include "ObjectExtrasTransform.glsl"

layout(location = 1) flat out uint ObjectId;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.Index.Slot].Indices[draw.Index.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldTransform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;

    gl_Position = SceneViewUBO.ViewProj * vec4(ObjectExtrasWorldPos(draw, vert, world, idx), 1.0);
}
