#version 450

#include "Bindless.glsl"

layout(location = 0) flat out uint InstanceIndex;

void main() {
    InstanceIndex = gl_InstanceIndex;
    const uint idx = IndexBuffers[pc.IndexSlot].Indices[gl_VertexIndex];
    const vec3 position = VertexBuffers[pc.VertexSlot].Vertices[idx + pc.VertexOffset].Position;
    const mat4 model = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex].M;
    gl_Position = Scene.Proj * Scene.View * model * vec4(position, 1.0);
}
