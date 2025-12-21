#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

void main() {
    const uint idx = IndexBuffers[nonuniformEXT(pc.IndexSlot)].Indices[gl_VertexIndex];
    const vec3 position = VertexBuffers[nonuniformEXT(pc.VertexSlot)].Vertices[idx].Position;
    const mat4 model = ModelBuffers[nonuniformEXT(pc.ModelSlot)].Models[pc.FirstInstance + gl_InstanceIndex].M;
    gl_Position = Scene.Proj * Scene.View * model * vec4(position, 1.0);
}
