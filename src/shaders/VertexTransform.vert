#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;
layout(location = 3) flat out uint FaceOverlayFlags;

layout(constant_id = 0) const uint OverlayKind = 0u;

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.Index.Slot].Indices[draw.Index.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const WorldMatrix world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint element_state = 0u;
    uint face_id = 0u;
    vec3 normal = vert.Normal;
    const vec3 morphed_pos = ApplyMorphDeform(draw, vert.Position, idx);
    const vec3 local_pos = ApplyArmatureDeform(draw, morphed_pos, idx, normal);
    if (draw.ObjectIdSlot != INVALID_SLOT) {
        face_id = ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FaceIdOffset + uint(gl_VertexIndex) / 3u];
        if (draw.FaceNormal.Slot != INVALID_SLOT && face_id != 0u) {
            normal = FaceNormalBuffers[draw.FaceNormal.Slot].Normals[draw.FaceNormal.Offset + face_id - 1u];
        }
        if (draw.ElementState.Slot != INVALID_SLOT && face_id != 0u) {
            element_state = uint(ElementStateBuffers[draw.ElementState.Slot].States[draw.ElementState.Offset + face_id - 1u]);
        }
    } else if (draw.ElementState.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementState.Slot].States[draw.ElementState.Offset + gl_VertexIndex]);
    }
    vec3 world_pos = vec3(world.M * vec4(local_pos, 1.0));
    if (should_apply_pending_transform(draw, idx)) {
        world_pos = apply_pending_transform(local_pos, world_pos, world, draw);
    }

    WorldNormal = mat3(world.MInv) * normal;
    WorldPosition = world_pos;

    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionModeEdit;
    const bool is_edit_edge = is_edit_mode && SceneViewUBO.EditElement == EditElementEdge;
    const vec4 edge_color = is_edit_mode ? vec4(ViewportTheme.Colors.WireEdit, 1.0) : vec4(ViewportTheme.Colors.Wire, 1.0);
    const vec4 object_base_color = vec4(0.7, 0.7, 0.7, 1);
    const vec4 base_color = draw.ObjectIdSlot != INVALID_SLOT ? object_base_color :
        draw.ElementState.Slot != INVALID_SLOT ? edge_color :
        OverlayKind == 1u ? vec4(ViewportTheme.Colors.FaceNormal, 1.0) :
        OverlayKind == 2u ? vec4(ViewportTheme.Colors.VertexNormal, 1.0) :
                            edge_color;
    const bool is_face_draw = draw.ObjectIdSlot != INVALID_SLOT;
    const bool is_edge_draw = !is_face_draw && draw.ElementState.Slot != INVALID_SLOT;
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    FaceOverlayFlags = 0u;

    vec4 selected_color = base_color;
    if (is_selected && is_edge_draw) {
        selected_color = is_edit_edge ?
            vec4(ViewportTheme.Colors.EdgeSelected, 1.0) :
            vec4(ViewportTheme.Colors.EdgeSelectedIncidental, 1.0);
    }

    if (is_face_draw) {
        if (is_selected) FaceOverlayFlags |= 1u;
        if (is_active) FaceOverlayFlags |= 2u;
        Color = base_color;
    } else {
        vec4 final_color = is_selected ? selected_color : base_color;
        if (is_active) final_color = vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0);
        Color = final_color;
    }
    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
}
