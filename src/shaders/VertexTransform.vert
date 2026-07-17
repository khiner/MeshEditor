#version 450

#include "Bindless.glsl"
#include "MorphDeform.glsl"
#include "ArmatureDeform.glsl"
#include "TransformUtils.glsl"

layout(location = 0) out vec3 WorldNormal;
layout(location = 1) out vec3 WorldPosition;
layout(location = 2) out vec4 Color;
layout(location = 3) flat out uint FaceOverlayFlags;
layout(location = 4) out vec2 TexCoord0;
layout(location = 5) out vec2 TexCoord1;
layout(location = 6) out vec2 TexCoord2;
layout(location = 7) out vec2 TexCoord3;
layout(location = 8) flat out uint MaterialIndex;
layout(location = 9) out vec4 VertexColor;
layout(location = 10) out vec4 WorldTangent;
layout(location = 11) flat out float WorldScale;
// Screen-space edge endpoints for line anti-aliasing (only used when rendering as LINE_LIST).
// edge_start is flat (constant across the primitive, takes value from the provoking vertex = first vertex).
// edge_pos is smooth (interpolated), giving the current position along the line.
layout(location = 12) flat out vec2 EdgeStart;
layout(location = 13) out vec2 EdgePos;
#ifdef VELOCITY_OUTPUT
// Screen motion across the shutter, as (prev->current, current->next) in UV space.
// Both halves point the same way, so the gather can walk either direction with one vector.
layout(location = 14) out vec4 Motion;
#endif

layout(constant_id = 0) const uint OverlayKind = 0u;
layout(constant_id = 1) const uint IsLineDraw = 0u;

vec3 ComputeWorldPos(DrawData draw, Transform world, uint vertex_index) {
    vec3 pos = VertexBuffers[draw.VertexSlot].Vertices[vertex_index + draw.VertexOffset].Position;
    vec3 n = vec3(0);
    ApplyMorphDeform(draw, pos, vertex_index, n);
    pos = ApplyArmatureDeform(draw, pos, vertex_index, n);
    return apply_pending_transform(draw, world, pos, vertex_index);
}

#ifdef VELOCITY_OUTPUT
// World position of `vert` under one pose. The per-draw offsets are shared across poses,
// so a pose is selected purely by its buffer slots.
vec3 PoseWorldPos(DrawData draw, Vertex vert, uint idx, uint model_slot, uint armature_slot, uint morph_slot) {
    vec3 normal = vert.Normal;
    vec3 pos = vert.Position;
    ApplyMorphDeform(draw, pos, idx, normal, morph_slot);
    const vec3 local_pos = ApplyArmatureDeform(draw, pos, idx, normal, armature_slot);
    const Transform world = ModelBuffers[nonuniformEXT(model_slot)].Models[draw.FirstInstance];
    return apply_pending_transform(draw, world, local_pos, idx);
}

// Each pose projects through its own view: looking through an animated camera moves the view
// across the shutter too, and a pure camera move is motion like any other.
vec2 ProjectToUv(mat4 view_proj, vec3 world_pos) {
    const vec4 clip = view_proj * vec4(world_pos, 1.0);
    return clip.xy / clip.w;
}
#endif

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    const Transform world = ModelBuffers[draw.ModelSlot].Models[draw.FirstInstance];

    uint element_state = 0u;
    uint face_id = 0u;
    MaterialIndex = 0u;
    vec3 normal = vert.Normal;
    vec3 morphed_pos = vert.Position;
    ApplyMorphDeform(draw, morphed_pos, idx, normal);
    const vec3 local_pos = ApplyArmatureDeform(draw, morphed_pos, idx, normal);
    bool computed_face_normal = false;
    if (draw.ObjectIdSlot != INVALID_SLOT) {
        face_id = ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FaceIdOffset + uint(gl_VertexIndex) / 3u];
        if (draw.FaceFirstTriOffset != INVALID_OFFSET && face_id != 0u) {
            // Compute full world position (morph, armature, pending transform) for all 3 verts of the first triangle of this face,
            // to compute flat face normals from the first triangle's vertices.
            // This duplicates per-vertex transform work (~3x for flat-shaded faces).
            // These shenanigans won't be needed with mesh shaders, which have per-primitive outputs.
            const uint first_tri = ObjectIdBuffers[SceneViewUBO.FaceFirstTriSlot].Ids[draw.FaceFirstTriOffset + face_id - 1u];
            const uint base = draw.IndexSlotOffset.Offset + first_tri * 3u;
            const uint i0 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base];
            const uint i1 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base + 1u];
            const uint i2 = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[base + 2u];
            const vec3 p0 = ComputeWorldPos(draw, world, i0);
            const vec3 p1 = ComputeWorldPos(draw, world, i1);
            const vec3 p2 = ComputeWorldPos(draw, world, i2);
            WorldNormal = normalize(cross(p1 - p0, p2 - p0));
            computed_face_normal = true;
        }
        if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT && face_id != 0u) {
            element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + face_id - 1u]);
        }
    } else if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + gl_VertexIndex]);
    }
    vec3 world_pos = apply_pending_transform(draw, world, local_pos, idx);

    if (!computed_face_normal) {
        WorldNormal = trs_transform_normal(world, normal);
    }
    WorldPosition = world_pos;
    VertexColor = vert.Color;

    const bool is_edit_mode = SceneViewUBO.InteractionMode == InteractionMode_Edit;
    const bool is_edit_edge = is_edit_mode && SceneViewUBO.EditElement == Element_Edge;
    const vec4 edge_color = is_edit_mode ? vec4(ViewportTheme.Colors.WireEdit, 1.0) : vec4(ViewportTheme.Colors.Wire, 1.0);
    const vec4 object_base_color = vec4(0.8, 0.8, 0.8, 1); // Blender's View3DShading.single_color default
    const vec4 base_color = draw.ObjectIdSlot != INVALID_SLOT ? object_base_color :
        draw.ElementStateSlotOffset.Slot != INVALID_SLOT ? edge_color :
        OverlayKind == 1u ? vec4(ViewportTheme.Colors.FaceNormal, 1.0) :
        OverlayKind == 2u ? vec4(ViewportTheme.Colors.VertexNormal, 1.0) :
                            edge_color;
    const bool is_face_draw = draw.ObjectIdSlot != INVALID_SLOT;
    const bool is_edge_draw = !is_face_draw && draw.ElementStateSlotOffset.Slot != INVALID_SLOT;
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
        if (draw.FacePrimitiveOffset != INVALID_OFFSET && draw.PrimitiveMaterialOffset != INVALID_OFFSET && face_id != 0u) {
            const uint primitive_index = FacePrimitiveBuffers[nonuniformEXT(SceneViewUBO.FacePrimitiveSlot)].PrimitiveIndices[draw.FacePrimitiveOffset + face_id - 1u];
            MaterialIndex = PrimitiveMaterialBuffers[nonuniformEXT(SceneViewUBO.PrimitiveMaterialSlot)].MaterialIndices[draw.PrimitiveMaterialOffset + primitive_index];
        }
        if (is_selected) FaceOverlayFlags |= 1u;
        if (is_active) FaceOverlayFlags |= 2u;
        Color = base_color;
    } else {
        vec4 final_color = is_selected ? selected_color : base_color;
        const bool is_excited = (element_state & STATE_EXCITED) != 0u;
        if (is_excited) final_color = ViewportTheme.Colors.ElementExcited;
        else if (is_active) final_color = vec4(ViewportTheme.Colors.ElementActive.rgb, 1.0);
        Color = final_color;
    }
    TexCoord0 = vert.TexCoord0;
    TexCoord1 = vert.TexCoord1;
    TexCoord2 = vert.TexCoord2;
    TexCoord3 = vert.TexCoord3;
    {
        vec3 tangent = vert.Tangent.xyz;
        if (dot(tangent, tangent) > 1e-8) {
            tangent = normalize(tangent);
            vec3 tangent_dummy_pos = vec3(0.0);
            ApplyArmatureDeform(draw, tangent_dummy_pos, idx, tangent);
            tangent = normalize(trs_transform_normal(world, tangent));
            WorldTangent = vec4(tangent, vert.Tangent.w);
        } else {
            WorldTangent = vec4(0, 0, 0, 1);
        }
    }
    WorldScale = (world.S.x + world.S.y + world.S.z) / 3.0;
    gl_Position = SceneViewUBO.ViewProj * vec4(world_pos, 1.0);
#ifdef VELOCITY_OUTPUT
    {
        const vec3 prev = PoseWorldPos(draw, vert, idx, SceneViewUBO.PrevModelSlot, SceneViewUBO.PrevArmatureDeformSlot, SceneViewUBO.PrevMorphWeightsSlot);
        const vec3 next = PoseWorldPos(draw, vert, idx, SceneViewUBO.NextModelSlot, SceneViewUBO.NextArmatureDeformSlot, SceneViewUBO.NextMorphWeightsSlot);
        const vec2 curr_uv = ProjectToUv(SceneViewUBO.ViewProj, world_pos);
        const vec2 prev_uv = ProjectToUv(SceneViewUBO.PrevViewProj, prev);
        const vec2 next_uv = ProjectToUv(SceneViewUBO.NextViewProj, next);
        // NDC spans 2 units across the viewport, so halving converts these to UV. The second half is
        // stored pointing backward like the first, which the gather's motion scale undoes.
        Motion = vec4(prev_uv - curr_uv, curr_uv - next_uv) * 0.5;
    }
#endif
    if (IsLineDraw != 0u) {
        gl_Position.z -= NdcOffsetFactor() * 1.0; // Push lines in front of faces (Blender: edge_ndc_offset_)
        // Convert clip-space to pixel coordinates matching gl_FragCoord.xy ([0, ViewportSize])
        const vec2 screen_pos = (gl_Position.xy / gl_Position.w * 0.5 + 0.5) * SceneViewUBO.ViewportSize;
        EdgeStart = screen_pos; // flat: takes first-vertex value for the whole line primitive
        EdgePos = screen_pos;   // smooth: interpolated along the line
    }
}
