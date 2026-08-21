#ifndef VERTEXTRANSFORM_MSL
#define VERTEXTRANSFORM_MSL

#include "Bindless.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "CornerClass.metal"
#include "CornerClassEncoding.metal"
#include "LineQuad.metal"
#include "MorphDeform.metal"
#include "ArmatureDeform.metal"
#include "TransformUtils.metal"
#include "MainDrawPushConstants.metal"
// LineQuad widens each line into a screen-space quad, six vertices per line.
// VelocityOutput writes the shutter-open and shutter-close motion the velocity pass reads.
#include "MeshVertexConstant.metal"

// World position of `vert` under one pose. The per-draw offsets are shared across poses,
// so a pose is selected purely by its buffer slots.
inline float3 PoseWorldPos(const thread Scene &scene, DrawData draw, Vertex vert, uint idx, uint model_slot, uint armature_slot, uint morph_slot) {
    float3 normal = float3(0);
    float3 pos = float3(vert.Position);
    ApplyMorphDeform(scene, draw, pos, idx, morph_slot);
    const float3 local_pos = ApplyArmatureDeform(scene, draw, pos, idx, normal, armature_slot);
    const Transform world = scene.Models(model_slot)[draw.FirstInstance];
    return trs_transform_point(world, local_pos);
}

// Rotate the derived corner normal by the stored (polar, azimuth) offset.
// The corner frame mirrors MeshStore::ComputeCornerFrame and must stay in lockstep with it.
// Its axes: the derived normal, the first non-degenerate outgoing edge projected off it, and their cross.
// A fixed axis substitutes for a zero normal, and a fixed perpendicular when both edges are degenerate.
// The frame rebuilds from current local positions, so authored offsets ride every deformation.
inline float3 ApplyNormalOffset(const thread Scene &scene, DrawData draw, uint vertex_id, float3 normal, float2 offset) {
    const float3 n = dot(normal, normal) > 0.0f ? normal : float3(0, 0, 1);
    const uint tri = (vertex_id / 3u) * 3u;
    const uint k = vertex_id - tri;
    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const float3 p0 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + k]);
    // An edge nearly parallel to the normal rejects to cancellation noise, so its perpendicular part
    // must be a meaningful fraction of its length to anchor the frame.
    // The two edge candidates stay unrolled: a loop here costs ~5% frame time on vertex-bound scenes.
    float3 ref;
    const float3 e1 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + (k + 1u) % 3u]) - p0;
    const float3 r1 = e1 - n * dot(e1, n);
    const float l1 = length(r1);
    if (l1 > 1e-3f * length(e1)) {
        ref = r1 / l1;
    } else {
        const float3 e2 = scene.GetLocalPosition(draw, indices[draw.IndexSlotOffset.Offset + tri + (k + 2u) % 3u]) - p0;
        const float3 r2 = e2 - n * dot(e2, n);
        const float l2 = length(r2);
        if (l2 > 1e-3f * length(e2)) {
            ref = r2 / l2;
        } else {
            const float3 axis = abs(n.x) < 0.5f ? float3(1, 0, 0) : float3(0, 1, 0);
            ref = normalize(cross(n, axis));
        }
    }
    const float3 ortho = cross(n, ref);
    return cos(offset.x) * n + sin(offset.x) * (cos(offset.y) * ref + sin(offset.y) * ortho);
}

// Corner shading normal for a face draw, composed from the corner classification.
// Vertex corners read the smooth vertex normal, Face the face normal, and Seam the sector normal.
// Each reads the posed store when derived this frame, else the base store.
// Authored corner-normal offsets apply wherever they are non-identity.
// Authored morph shading adds the targets' weighted normal deltas at the end.
// A uniform mesh stores no class buffer, so the offset value itself carries the class.
inline float3 CornerNormal(const thread Scene &scene, DrawData draw, uint vertex_id, uint idx, uint face_id) {
    const uint value = draw.CornerClassOffset == INVALID_OFFSET ? CornerClass_Vertex << CornerClassEncoding_TagShift :
        draw.CornerClassOffset == CornerClassEncoding_UniformFaceOffset ? CornerClass_Face << CornerClassEncoding_TagShift :
                                                                          scene.CornerClasses(scene.View.CornerClassSlot)[draw.CornerClassOffset + vertex_id];
    const uint tag = value >> CornerClassEncoding_TagShift;
    float3 normal;
    if (tag == CornerClass_Vertex) {
        normal = scene.GetVertexNormal(draw, idx);
    } else if (tag == CornerClass_Face) {
        normal = draw.PosedFaceNormalOffset != INVALID_OFFSET ?
            float3(scene.PosedFaceNormals(scene.View.PosedFaceNormalSlot)[draw.PosedFaceNormalOffset + face_id - 1u]) :
            float3(scene.BaseFaceNormals(scene.View.BaseFaceNormalSlot)[draw.BaseFaceNormalOffset + face_id - 1u]);
    } else {
        const uint seam = value & CornerClassEncoding_IndexMask;
        normal = draw.PosedSeamNormalOffset != INVALID_OFFSET ?
            float3(scene.PosedSeamNormals(scene.View.PosedSeamNormalSlot)[draw.PosedSeamNormalOffset + seam]) :
            float3(scene.BaseSeamNormals(scene.View.BaseSeamNormalSlot)[draw.BaseSeamNormalOffset + seam]);
    }
    if (draw.CustomCornerMaskOffset != INVALID_OFFSET) {
        const uint corner = draw.CornerBase + vertex_id;
        const uint2 mask = uint2(scene.CustomCornerMasks(scene.View.CustomCornerMaskSlot)[draw.CustomCornerMaskOffset + corner / 32u]);
        const uint bit = 1u << (corner % 32u);
        if ((mask.x & bit) != 0u) {
            const uint packed = mask.y + popcount(mask.x & (bit - 1u));
            const float2 offset = float2(scene.CustomCornerNormals(scene.View.CustomCornerNormalSlot)[draw.CustomCornerNormalOffset + packed]);
            normal = ApplyNormalOffset(scene, draw, vertex_id, normal, offset);
        }
    }
    // glTF morphed normal: normalize(N0 + sum(w_t * NormalDelta_t)).
    // The pose pre-pass accumulates the per-vertex delta sum.
    // Authored-morph draws carry no posed normal offsets, so the branches above composed the rest normals.
    if (draw.MorphShadingAuthored != 0u) {
        normal = NormalizeOrZero(normal + float3(scene.PosedMorphNormalDeltas(scene.View.PosedMorphNormalDeltaSlot)[draw.PosedPositionOffset + idx]));
    }
    return normal;
}

inline MeshVaryings TransformVertex(const thread Scene &scene, DrawData draw, uint vertex_id, uint vertex_index, uint idx) {
    MeshVaryings out;
    const uint corner = LineQuad ? line_quad_corner(vertex_id) : 0u;
    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    // Motion blur steps read their captured transforms through the override, keeping DrawData step-agnostic.
    const uint model_slot = scene.View.ModelSlotOverride != INVALID_SLOT ? scene.View.ModelSlotOverride : draw.ModelSlot;
    const Transform world = scene.Models(model_slot)[draw.FirstInstance];

    uint element_state = 0u;
    uint face_id = 0u;
    out.MaterialIndex = 0u;
    const float3 local_pos = scene.GetLocalPosition(draw, idx);
    // Face draws compose the corner shading normal.
    // Point and line draws shade from the vertex normal.
    const bool is_face_draw = draw.ObjectIdSlot != INVALID_SLOT;
    float3 normal = is_face_draw ? float3(0) : scene.GetVertexNormal(draw, idx);
    if (is_face_draw) {
        face_id = scene.ObjectIds(draw.ObjectIdSlot)[draw.FaceIdOffset + vertex_index / 3u];
        normal = CornerNormal(scene, draw, vertex_id, idx, face_id);
        if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT && face_id != 0u) {
            element_state = uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + face_id - 1u]);
        }
    } else if (draw.ElementStateSlotOffset.Slot != INVALID_SLOT) {
        element_state = uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + vertex_index]);
    }
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, local_pos));

    out.WorldNormal = trs_transform_normal(world, normal);
    out.WorldPosition = world_pos;
    // Face draws hold one color per corner, point and line draws one per vertex.
    const uint color_index = is_face_draw ? vertex_index : idx;
    out.VertexColor = draw.CornerColorOffset != INVALID_OFFSET ?
        float4(scene.CornerColors(scene.View.CornerColorSlot)[draw.CornerColorOffset + color_index]) :
        float4(1.0f);

    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const bool is_edit_mode = scene.View.InteractionMode == InteractionMode_Edit;
    const bool is_edit_edge = is_edit_mode && scene.View.EditElement == Element_Edge;
    const float4 edge_color = is_edit_mode ? float4(float3(colors.WireEdit), 1.0f) : float4(float3(colors.Wire), 1.0f);
    const float4 object_base_color = float4(0.8f, 0.8f, 0.8f, 1.0f); // Blender's View3DShading.single_color default
    const float4 base_color = draw.ObjectIdSlot != INVALID_SLOT ? object_base_color :
        draw.ElementStateSlotOffset.Slot != INVALID_SLOT ? edge_color :
        OverlayKind == 1u ? float4(float3(colors.FaceNormal), 1.0f) :
        OverlayKind == 2u ? float4(float3(colors.VertexNormal), 1.0f) :
                            edge_color;
    const bool is_edge_draw = !is_face_draw && draw.ElementStateSlotOffset.Slot != INVALID_SLOT;
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    out.FaceOverlayFlags = 0u;

    float4 selected_color = base_color;
    if (is_selected && is_edge_draw) {
        selected_color = is_edit_edge ?
            float4(float3(colors.EdgeSelected), 1.0f) :
            float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    }

    // Face draws index the primitive table per face, point and line draws per vertex.
    if (draw.ElementPrimitiveOffset != INVALID_OFFSET && draw.PrimitiveMaterialOffset != INVALID_OFFSET && (!is_face_draw || face_id != 0u)) {
        const uint element = is_face_draw ? face_id - 1u : idx;
        const uint primitive_index = scene.ElementPrimitives(scene.View.ElementPrimitiveSlot)[draw.ElementPrimitiveOffset + element];
        out.MaterialIndex = scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[draw.PrimitiveMaterialOffset + primitive_index];
    }
    if (is_face_draw) {
        if (is_selected) out.FaceOverlayFlags |= 1u;
        if (is_active) out.FaceOverlayFlags |= 2u;
        out.Color = base_color;
    } else if (is_edge_draw && scene.View.InteractionMode == InteractionMode_Object && scene.View.ShowOverlays != 0u) {
        out.Color = scene.ObjectSelectionColor(scene.InstanceState(draw), base_color);
    } else {
        float4 final_color = is_selected ? selected_color : base_color;
        const bool is_excited = (element_state & STATE_EXCITED) != 0u;
        if (is_excited) final_color = float4(colors.ElementExcited);
        else if (is_active) final_color = float4(float4(colors.ElementActive).rgb, 1.0f);
        out.Color = final_color;
    }
    const uint corner_uv_slot = scene.View.CornerUvSlot;
    device const packed_float2 *uvs = scene.CornerUvs(corner_uv_slot);
    out.TexCoord0 = draw.CornerUvOffsets[0] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[0] + vertex_index]) : float2(0);
    out.TexCoord1 = draw.CornerUvOffsets[1] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[1] + vertex_index]) : float2(0);
    out.TexCoord2 = draw.CornerUvOffsets[2] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[2] + vertex_index]) : float2(0);
    out.TexCoord3 = draw.CornerUvOffsets[3] != INVALID_OFFSET ? float2(uvs[draw.CornerUvOffsets[3] + vertex_index]) : float2(0);
    {
        const float4 vertex_tangent = draw.CornerTangentOffset != INVALID_OFFSET ?
            float4(scene.CornerTangents(scene.View.CornerTangentSlot)[draw.CornerTangentOffset + vertex_index]) :
            float4(0, 0, 0, 1);
        float3 tangent = vertex_tangent.xyz;
        if (dot(tangent, tangent) > 1e-8f) {
            tangent = normalize(tangent);
            float3 tangent_dummy_pos = float3(0.0f);
            ApplyArmatureDeform(scene, draw, tangent_dummy_pos, idx, tangent);
            tangent = normalize(trs_transform_normal(world, tangent));
            out.WorldTangent = float4(tangent, vertex_tangent.w);
        } else {
            out.WorldTangent = float4(0, 0, 0, 1);
        }
    }
    const float3 world_scale = float3(world.S);
    out.WorldScale = (world_scale.x + world_scale.y + world_scale.z) / 3.0f;
    out.Position = scene.ViewProj() * float4(world_pos, 1.0f);
    out.PointSize = PointSize;
    if (LineQuad) {
        // The line's other endpoint, transformed through the same deformations.
        const uint other_index = (vertex_id / 6u) * 2u + 1u - line_quad_endpoint(corner);
        const uint other_idx = indices[draw.IndexSlotOffset.Offset + other_index];
        const float3 other_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, scene.GetLocalPosition(draw, other_idx)));
        const float4 other_clip = scene.ViewProj() * float4(other_pos, 1.0f);
        // Half-width matches the wire overlay: its core plus the anti-aliased fringe.
        const bool first = line_quad_endpoint(corner) == 0u;
        out.Position = line_quad_position(scene, first ? out.Position : other_clip, first ? other_clip : out.Position, corner, scene.Theme.EdgeWidth + 0.5f);
    }
    out.MotionPrev = float3(0);
    out.MotionNext = float3(0);
    if (VelocityOutput) {
        const float3 prev = PoseWorldPos(scene, draw, vert, idx, scene.View.PrevModelSlot, scene.View.PrevArmatureDeformSlot, scene.View.PrevMorphWeightsSlot);
        const float3 next = PoseWorldPos(scene, draw, vert, idx, scene.View.NextModelSlot, scene.View.NextArmatureDeformSlot, scene.View.NextMorphWeightsSlot);
        out.MotionPrev = prev - world_pos;
        out.MotionNext = next - world_pos;
    }
    out.EdgeStart = float2(0);
    out.EdgePos = float2(0);
    if (IsLineDraw != 0u) {
        out.Position.z -= NdcOffsetFactor(scene) * 1.0f; // Push lines in front of faces.
        const float2 screen_pos = clip_to_frag_co(out.Position, float2(scene.View.ViewportSize));
        out.EdgeStart = screen_pos; // flat: takes the first vertex's value for the whole line primitive
        out.EdgePos = screen_pos;   // smooth: interpolated along the line
    }
    return out;
}

vertex MeshVaryings VertexTransformVertex(
    uint vertex_id [[vertex_id]],
    uint instance_id [[instance_id]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant MainDrawPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const DrawData draw = GetDrawData(scene, pc.DrawDataOffset, instance_id);
    const uint corner = LineQuad ? line_quad_corner(vertex_id) : 0u;
    const uint vertex_index = LineQuad ? (vertex_id / 6u) * 2u + line_quad_endpoint(corner) : vertex_id;
    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    return TransformVertex(scene, draw, vertex_id, vertex_index, indices[draw.IndexSlotOffset.Offset + vertex_index]);
}

#endif
