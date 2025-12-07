#version 450
#extension GL_EXT_nonuniform_qualifier : require

#include "Bindless.glsl"

void main() {
    const uint idx = IndexBuffers[nonuniformEXT(pc.IndexSlot)].indices[pc.FirstIndex + gl_VertexIndex] + pc.VertexOffset;
    const vec3 position = VertexBuffers[nonuniformEXT(pc.VertexSlot)].vertices[idx].Position;
    const mat4 model = ModelBuffers[nonuniformEXT(pc.ModelSlot)].models[pc.FirstInstance].M;
    gl_Position = Scene.Proj * Scene.View * model * vec4(position, 1.0);
}
