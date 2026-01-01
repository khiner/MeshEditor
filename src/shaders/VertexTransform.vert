#version 450

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

void main() {
    const uint idx = IndexBuffers[pc.IndexSlot].Indices[pc.IndexOffset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[pc.VertexSlot].Vertices[idx + pc.VertexOffset];
    const WorldMatrix world = ModelBuffers[pc.ModelSlot].Models[pc.FirstInstance + gl_InstanceIndex];

    uint state = 0u;
    uint face_id = 0u;
    vec3 normal = vert.Normal;
    if (pc.ObjectIdSlot != INVALID_SLOT) {
        face_id = ObjectIdBuffers[pc.ObjectIdSlot].Ids[gl_VertexIndex + pc.FaceIdOffset];
        if (pc.FaceNormalSlot != INVALID_SLOT && face_id != 0u) {
            normal = FaceNormalBuffers[pc.FaceNormalSlot].Normals[pc.FaceNormalOffset + face_id - 1u];
        }
        if (pc.ElementStateSlot != INVALID_SLOT && face_id != 0u) {
            state = ElementStateBuffers[pc.ElementStateSlot].States[face_id - 1u];
        }
    } else if (pc.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[pc.ElementStateSlot].States[gl_VertexIndex];
    }

    WorldNormal = mat3(world.MInv) * normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    vec4 base_color = pc.LineColor;
    if (pc.ObjectIdSlot != INVALID_SLOT) {
        base_color = Scene.BaseColor;
    } else if (pc.ElementStateSlot != INVALID_SLOT) {
        base_color = Scene.EdgeColor;
    }
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    Color = is_active ? Scene.ActiveColor : is_selected ? Scene.SelectedColor : base_color;
    gl_Position = Scene.Proj * Scene.View * world.M * vec4(vert.Position, 1.0);
}
