#version 450

#define DRAW_DATA_INDEX gl_InstanceIndex
#include "Bindless.glsl"

layout(location = 1) flat out uint ObjectId;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlot].Indices[draw.IndexOffset + uint(gl_VertexIndex)];
    const vec3 position = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset].Position;
    const mat4 model = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance].M;
    ObjectId = (draw.ObjectIdSlot != INVALID_SLOT) ? ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FirstInstance] : 0u;
    gl_Position = Scene.Proj * Scene.View * model * vec4(position, 1.0);
}
