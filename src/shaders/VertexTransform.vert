#version 450

#include "Bindless.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;

layout(constant_id = 0) const uint OverlayKind = 0u;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlot].Indices[draw.IndexOffset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint state = 0u;
    uint face_id = 0u;
    vec3 normal = vert.Normal;
    if (draw.ObjectIdSlot != INVALID_SLOT) {
        face_id = ObjectIdBuffers[draw.ObjectIdSlot].Ids[gl_VertexIndex + draw.FaceIdOffset];
        if (draw.FaceNormalSlot != INVALID_SLOT && face_id != 0u) {
            normal = FaceNormalBuffers[draw.FaceNormalSlot].Normals[draw.FaceNormalOffset + face_id - 1u];
        }
        if (draw.ElementStateSlot != INVALID_SLOT && face_id != 0u) {
            state = ElementStateBuffers[draw.ElementStateSlot].States[face_id - 1u];
        }
    } else if (draw.ElementStateSlot != INVALID_SLOT) {
        state = ElementStateBuffers[draw.ElementStateSlot].States[gl_VertexIndex];
    }

    WorldNormal = mat3(world.MInv) * normal;
    WorldPosition = vec3(world.M * vec4(vert.Position, 1.0));
    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionModeEdit;
    const vec4 edge_color = is_edit_mode ? ViewportTheme.Colors.WireEdit : ViewportTheme.Colors.Wire;
    const vec4 object_base_color = vec4(0.7, 0.7, 0.7, 1);
    const vec4 base_color = draw.ObjectIdSlot != INVALID_SLOT ? object_base_color :
        draw.ElementStateSlot != INVALID_SLOT ? edge_color :
        OverlayKind == 1u ? ViewportTheme.Colors.FaceNormal :
        OverlayKind == 2u ? ViewportTheme.Colors.VertexNormal :
                            edge_color;
    const bool is_selected = (state & STATE_SELECTED) != 0u;
    const bool is_active = (state & STATE_ACTIVE) != 0u;
    Color = is_active ? ViewportTheme.Colors.ElementActive : is_selected ? ViewportTheme.Colors.ElementSelected : base_color;
    gl_Position = SceneViewUBO.Proj * SceneViewUBO.View * world.M * vec4(vert.Position, 1.0);
}
