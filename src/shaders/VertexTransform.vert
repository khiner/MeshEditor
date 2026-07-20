#version 450

#include "Bindless.glsl"
#include "CornerClass.glsl"
#include "CornerClassEncoding.glsl"
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
// World-space motion across the shutter, as offsets from the current position. World deltas
// interpolate exactly across a triangle, so the fragment shader projects them per pixel.
layout(location = 14) out vec3 MotionPrev;
layout(location = 15) out vec3 MotionNext;
#endif

layout(constant_id = 0) const uint OverlayKind = 0u;
layout(constant_id = 1) const uint IsLineDraw = 0u;

#ifdef VELOCITY_OUTPUT
// World position of `vert` under one pose. The per-draw offsets are shared across poses,
// so a pose is selected purely by its buffer slots.
vec3 PoseWorldPos(DrawData draw, Vertex vert, uint idx, uint model_slot, uint armature_slot, uint morph_slot) {
    vec3 normal = vec3(0);
    vec3 pos = vert.Position;
    ApplyMorphDeform(draw, pos, idx, morph_slot);
    const vec3 local_pos = ApplyArmatureDeform(draw, pos, idx, normal, armature_slot);
    const Transform world = ModelBuffers[nonuniformEXT(model_slot)].Models[draw.FirstInstance];
    return trs_transform_point(world, local_pos);
}
#endif

// Rotate the derived corner normal by the stored (polar, azimuth) offset.
// The corner frame mirrors MeshStore::ComputeCornerFrame and must stay in lockstep with it.
// Its axes: the derived normal, the first non-degenerate outgoing edge projected off it, and their cross.
// A fixed axis substitutes for a zero normal, and a fixed perpendicular when both edges are degenerate.
// The frame rebuilds from current local positions, so authored offsets ride every deformation.
vec3 ApplyNormalOffset(DrawData draw, vec3 normal, vec2 offset) {
    const vec3 n = dot(normal, normal) > 0.0 ? normal : vec3(0, 0, 1);
    const uint tri = (uint(gl_VertexIndex) / 3u) * 3u;
    const uint k = uint(gl_VertexIndex) - tri;
    const vec3 p0 = GetLocalPosition(draw, IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + tri + k]);
    // An edge nearly parallel to the normal rejects to cancellation noise, so its perpendicular part must be a meaningful fraction of its length to anchor the frame.
    // The two edge candidates stay unrolled: a loop here costs ~5% frame time on vertex-bound scenes.
    vec3 ref;
    const vec3 e1 = GetLocalPosition(draw, IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + tri + (k + 1u) % 3u]) - p0;
    const vec3 r1 = e1 - n * dot(e1, n);
    const float l1 = length(r1);
    if (l1 > 1e-3 * length(e1)) {
        ref = r1 / l1;
    } else {
        const vec3 e2 = GetLocalPosition(draw, IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + tri + (k + 2u) % 3u]) - p0;
        const vec3 r2 = e2 - n * dot(e2, n);
        const float l2 = length(r2);
        if (l2 > 1e-3 * length(e2)) {
            ref = r2 / l2;
        } else {
            const vec3 axis = abs(n.x) < 0.5 ? vec3(1, 0, 0) : vec3(0, 1, 0);
            ref = normalize(cross(n, axis));
        }
    }
    const vec3 ortho = cross(n, ref);
    return cos(offset.x) * n + sin(offset.x) * (cos(offset.y) * ref + sin(offset.y) * ortho);
}

// Corner shading normal for a face draw, composed from the corner classification.
// Vertex corners read the smooth vertex normal, Face the face normal, and Seam the sector normal.
// Each reads the posed store when derived this frame, else the base store.
// Authored corner-normal offsets apply wherever they are non-identity.
// Authored morph shading adds the targets' weighted normal deltas at the end.
// A uniform mesh stores no class buffer, so the offset value itself carries the class.
vec3 CornerNormal(DrawData draw, uint idx, uint face_id) {
    const uint value = draw.CornerClassOffset == INVALID_OFFSET ? CornerClass_Vertex << CornerClassEncoding_TagShift :
        draw.CornerClassOffset == CornerClassEncoding_UniformFaceOffset ? CornerClass_Face << CornerClassEncoding_TagShift :
                                                                          CornerClassBuffers[nonuniformEXT(SceneViewUBO.CornerClassSlot)].Classes[draw.CornerClassOffset + uint(gl_VertexIndex)];
    const uint tag = value >> CornerClassEncoding_TagShift;
    vec3 normal;
    if (tag == CornerClass_Vertex) {
        normal = GetVertexNormal(draw, idx);
    } else if (tag == CornerClass_Face) {
        normal = draw.PosedFaceNormalOffset != INVALID_OFFSET ?
            PosedFaceNormalBuffers[nonuniformEXT(SceneViewUBO.PosedFaceNormalSlot)].Normals[draw.PosedFaceNormalOffset + face_id - 1u] :
            BaseFaceNormalBuffers[nonuniformEXT(SceneViewUBO.BaseFaceNormalSlot)].Normals[draw.BaseFaceNormalOffset + face_id - 1u];
    } else {
        const uint seam = value & CornerClassEncoding_IndexMask;
        normal = draw.PosedSeamNormalOffset != INVALID_OFFSET ?
            PosedSeamNormalBuffers[nonuniformEXT(SceneViewUBO.PosedSeamNormalSlot)].Normals[draw.PosedSeamNormalOffset + seam] :
            BaseSeamNormalBuffers[nonuniformEXT(SceneViewUBO.BaseSeamNormalSlot)].Normals[draw.BaseSeamNormalOffset + seam];
    }
    if (draw.CustomCornerMaskOffset != INVALID_OFFSET) {
        const uint corner = draw.CornerBase + uint(gl_VertexIndex);
        const uvec2 mask = CustomCornerMaskBuffers[nonuniformEXT(SceneViewUBO.CustomCornerMaskSlot)].WordRanks[draw.CustomCornerMaskOffset + corner / 32u];
        const uint bit = 1u << (corner % 32u);
        if ((mask.x & bit) != 0u) {
            const uint packed = mask.y + bitCount(mask.x & (bit - 1u));
            const vec2 offset = CustomCornerNormalBuffers[nonuniformEXT(SceneViewUBO.CustomCornerNormalSlot)].Offsets[draw.CustomCornerNormalOffset + packed];
            normal = ApplyNormalOffset(draw, normal, offset);
        }
    }
    // glTF morphed normal: normalize(N0 + sum(w_t * NormalDelta_t)).
    // The pose pre-pass accumulates the per-vertex delta sum.
    // Authored-morph draws carry no posed normal offsets, so the branches above composed the rest normals.
    if (draw.MorphShadingAuthored != 0u) {
        normal = NormalizeOrZero(normal + PosedMorphNormalDeltaBuffers[nonuniformEXT(SceneViewUBO.PosedMorphNormalDeltaSlot)].Deltas[draw.PosedPositionOffset + idx]);
    }
    return normal;
}

void main() {
    const DrawData draw = GetDrawData();
    const uint idx = IndexBuffers[draw.IndexSlotOffset.Slot].Indices[draw.IndexSlotOffset.Offset + uint(gl_VertexIndex)];
    const Vertex vert = VertexBuffers[draw.VertexSlot].Vertices[idx + draw.VertexOffset];
    // Motion blur steps read their captured transforms through the override, keeping DrawData step-agnostic.
    const uint model_slot = SceneViewUBO.ModelSlotOverride != INVALID_SLOT ? SceneViewUBO.ModelSlotOverride : draw.ModelSlot;
    const Transform world = ModelBuffers[model_slot].Models[draw.FirstInstance];

    uint element_state = 0u;
    uint face_id = 0u;
    MaterialIndex = 0u;
    const vec3 local_pos = GetLocalPosition(draw, idx);
    // Face draws compose the corner shading normal.
    // Line draw fragments ignore WorldNormal, so it stays zero.
    vec3 normal = vec3(0);
    if (draw.ObjectIdSlot != INVALID_SLOT) {
        face_id = ObjectIdBuffers[draw.ObjectIdSlot].Ids[draw.FaceIdOffset + uint(gl_VertexIndex) / 3u];
        normal = CornerNormal(draw, idx, face_id);
        if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT && face_id != 0u) {
            element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + face_id - 1u]);
        }
    } else if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        element_state = uint(ElementStateBuffers[draw.ElementStateSlotOffset.Slot].States[draw.ElementStateSlotOffset.Offset + gl_VertexIndex]);
    }
    const vec3 world_pos = apply_object_pending_transform(draw, trs_transform_point(world, local_pos));

    WorldNormal = trs_transform_normal(world, normal);
    WorldPosition = world_pos;
    // Face draws read triangle-mesh colors per corner.
    // Line/point meshes keep a point-domain color on the vertex.
    VertexColor = draw.CornerColorOffset != INVALID_OFFSET ?
        CornerColorBuffers[nonuniformEXT(SceneViewUBO.CornerColorSlot)].Colors[draw.CornerColorOffset + uint(gl_VertexIndex)] :
        vert.Color;

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
    const uint corner_uv_slot = SceneViewUBO.CornerUvSlot;
    TexCoord0 = draw.CornerUvOffsets[0] != INVALID_OFFSET ? CornerUvBuffers[nonuniformEXT(corner_uv_slot)].Uvs[draw.CornerUvOffsets[0] + uint(gl_VertexIndex)] : vec2(0);
    TexCoord1 = draw.CornerUvOffsets[1] != INVALID_OFFSET ? CornerUvBuffers[nonuniformEXT(corner_uv_slot)].Uvs[draw.CornerUvOffsets[1] + uint(gl_VertexIndex)] : vec2(0);
    TexCoord2 = draw.CornerUvOffsets[2] != INVALID_OFFSET ? CornerUvBuffers[nonuniformEXT(corner_uv_slot)].Uvs[draw.CornerUvOffsets[2] + uint(gl_VertexIndex)] : vec2(0);
    TexCoord3 = draw.CornerUvOffsets[3] != INVALID_OFFSET ? CornerUvBuffers[nonuniformEXT(corner_uv_slot)].Uvs[draw.CornerUvOffsets[3] + uint(gl_VertexIndex)] : vec2(0);
    {
        const vec4 vertex_tangent = draw.CornerTangentOffset != INVALID_OFFSET ?
            CornerTangentBuffers[nonuniformEXT(SceneViewUBO.CornerTangentSlot)].Tangents[draw.CornerTangentOffset + uint(gl_VertexIndex)] :
            vec4(0, 0, 0, 1);
        vec3 tangent = vertex_tangent.xyz;
        if (dot(tangent, tangent) > 1e-8) {
            tangent = normalize(tangent);
            vec3 tangent_dummy_pos = vec3(0.0);
            ApplyArmatureDeform(draw, tangent_dummy_pos, idx, tangent);
            tangent = normalize(trs_transform_normal(world, tangent));
            WorldTangent = vec4(tangent, vertex_tangent.w);
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
        MotionPrev = prev - world_pos;
        MotionNext = next - world_pos;
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
