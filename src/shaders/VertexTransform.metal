#ifndef VERTEXTRANSFORM_MSL
#define VERTEXTRANSFORM_MSL

#include "Bindless.metal"
#include "ConnectivityRead.metal"
#include "CornerNormalOffset.metal"
#include "SceneUBO.metal"
#include "Varyings.metal"
#include "gpu/CornerClass.h"
#include "gpu/CornerClassMode.h"
#include "MorphDeform.metal"
#include "ArmatureDeform.metal"
#include "TransformUtils.metal"
#include "EditSelection.metal"

// Rebuild from current local polygon positions so offsets follow deformation.
inline float3 ApplyNormalOffset(const thread Scene &scene, DrawData draw, uint vertex_id, float3 normal, float2 offset) {
    const uint h = vertex_id;
    const ConnectivityView conn{scene.B, draw.Connectivity, draw.FaceCount};
    const auto position = [&](uint corner) {
        const uint v = scene.Indices(draw.IndexSlotOffset.Slot)[corner] - draw.VertexOffset;
        return scene.GetLocalPosition(draw, v);
    };
    return DecodeNormalOffset(offset, ComputeCornerFrame(normal, position(h), position(conn.Next(h)), position(conn.Previous(h))));
}

// Returns a face corner's shading normal from its class, posed state, and optional coarse normal.
inline float3 CornerNormal(
    const thread Scene &scene, DrawData draw, uint vertex_id, uint idx, uint face_id,
    bool coarse, float3 coarse_normal
) {
    const bool mixed = draw.CornerClassMode == uint(CornerClassMode::Mixed);
    const uint source_face = mixed ? scene.CornerFace(draw, vertex_id) : InvalidOffset;
    const bool flat = draw.CornerClassMode == uint(CornerClassMode::UniformFace) ||
        (mixed && BindlessBuffer(uchar, scene.B.Buffer, scene.View.FaceSharpnessSlot)[source_face] != 0u);
    const uint root = mixed && !flat ? CornerSectorRoot(scene.B, scene.View.CornerSectors, vertex_id) : InvalidOffset;
    float3 normal;
    if (flat) {
        normal = coarse ? coarse_normal : scene.GetFaceNormal(draw, face_id);
    } else if (root == InvalidOffset) {
        normal = scene.GetVertexNormal(draw, idx);
    } else {
        const uint record = ElementAttributeIndex(scene.B, scene.View.NormalSectors, root);
        const uint posed = PoseAttributeIndex(scene.B, scene.View.PosedSectorNodesSlot, draw.SectorNamespace, record);
        normal = posed != InvalidOffset ?
            float3(BindlessBuffer(packed_float3, scene.B.Buffer, scene.View.PosedSectorValuesSlot)[posed]) :
            float3(BindlessBuffer(NormalSector, scene.B.Buffer, scene.View.NormalSectors.ValuesSlot)[record].Normal);
    }
    if (draw.CustomNormals.ValuesSlot != InvalidSlot) {
        const float2 offset = CustomNormalOffset(scene.B, draw.CustomNormals, vertex_id);
        if (offset.x >= 0.f) normal = ApplyNormalOffset(scene, draw, vertex_id, normal, offset);
    }
    // glTF morphed normal: normalize(N0 + sum(w_t * NormalDelta_t)).
    if (draw.MorphShadingAuthored != 0u) {
        normal = NormalizeOrZero(normal + float3(scene.PosedMorphNormalDeltas(scene.View.PosedMorphNormalDeltaSlot)[PoseAttributeIndex(scene.B, scene.View.PosedMorphNormalNodesSlot, draw.MorphNormalNamespace, draw.VertexOffset + idx)]));
    }
    return normal;
}

inline MeshVaryings TransformVertex(
    const thread Scene &scene, DrawData draw, uint vertex_id, uint vertex_index, uint idx,
    bool face_attributes = true, bool shading_normal = true,
    bool coarse = false, float3 coarse_normal = float3(0.0f)
) {
    const Vertex vert = scene.Vertices(draw.VertexSlot)[idx + draw.VertexOffset];
    // Motion-blur steps use captured transforms without modifying DrawData.
    const uint model_slot = scene.View.ModelSlotOverride != InvalidSlot ? scene.View.ModelSlotOverride : draw.ModelSlot;
    const Transform world = scene.Models(model_slot)[draw.FirstInstance];

    uint element_state = 0u;
    uint face_id = InvalidOffset;
    uint material_index = 0u;
    const float3 local_pos = scene.GetLocalPosition(draw, idx);
    const bool is_face_draw = draw.TriangleSlot != InvalidSlot;
    float3 normal = is_face_draw ? float3(0) : scene.GetVertexNormal(draw, idx);
    if (is_face_draw && (face_attributes || shading_normal)) {
        // Coarse-cluster corners have no source-face identity.
        if (!coarse) face_id = scene.CornerFace(draw, vertex_index);
        if (shading_normal) normal = CornerNormal(scene, draw, vertex_id, idx, face_id, coarse, coarse_normal);
        if (face_attributes && face_id != InvalidOffset) element_state = EditFaceState(scene, draw, face_id);
    } else if (draw.Selection.Summary.Slot != InvalidSlot) {
        element_state = EditEdgeEndpointState(scene, draw, vertex_index / 2u, idx);
    }
    const float3 world_pos = apply_object_pending_transform(scene, draw, trs_transform_point(world, local_pos));

    bool has_attributes = draw.CornerTangent.ValuesSlot != InvalidSlot || draw.CornerColor.ValuesSlot != InvalidSlot;
    for (uint set = 0u; set < 4u; ++set) has_attributes |= draw.CornerUvs[set].ValuesSlot != InvalidSlot;
    const uint attribute_handle = !has_attributes ? 0u : is_face_draw ? vertex_index : draw.VertexOffset + idx;

    constant ViewportThemeColors &colors = scene.Theme.Colors;
    const bool is_edit_mode = scene.View.InteractionMode == InteractionMode::Edit;
    const bool is_edit_edge = is_edit_mode && scene.View.EditElement == Element::Edge;
    const float4 edge_color = is_edit_mode ? float4(float3(colors.WireEdit), 1.0f) : float4(float3(colors.Wire), 1.0f);
    const float4 object_base_color = float4(0.8f, 0.8f, 0.8f, 1.0f); // Matches Blender's View3DShading.single_color default.
    const float4 base_color = draw.TriangleSlot != InvalidSlot ? object_base_color : edge_color;
    const bool is_edge_draw = !is_face_draw && draw.Selection.Summary.Slot != InvalidSlot;
    const bool is_selected = (element_state & STATE_SELECTED) != 0u;
    const bool is_active = (element_state & STATE_ACTIVE) != 0u;

    uint face_overlay_flags = 0u;

    float4 selected_color = base_color;
    if (is_selected && is_edge_draw) {
        selected_color = is_edit_edge ?
            float4(float3(colors.EdgeSelected), 1.0f) :
            float4(float3(colors.EdgeSelectedIncidental), 1.0f);
    }

    if (face_attributes && draw.ElementPrimitives.ValuesSlot != InvalidSlot && draw.PrimitiveMaterialOffset != InvalidOffset && (!is_face_draw || face_id != InvalidOffset)) {
        const uint element = is_face_draw ? face_id : draw.VertexOffset + idx;
        const uint primitive_index = scene.ElementPrimitive(draw.ElementPrimitives, element);
        material_index = scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[draw.PrimitiveMaterialOffset + primitive_index];
    }
    float4 color = base_color;
    if (is_face_draw) {
        if (face_attributes && is_selected) face_overlay_flags |= 1u;
        if (face_attributes && is_active) face_overlay_flags |= 2u;
    } else if (is_edge_draw && scene.View.InteractionMode == InteractionMode::Object && scene.View.ShowOverlays != 0u) {
        color = scene.ObjectSelectionColor(scene.InstanceState(draw), base_color);
    } else {
        float4 final_color = is_selected ? selected_color : base_color;
        if (is_active) final_color = float4(float4(colors.ElementActive).rgb, 1.0f);
        color = final_color;
    }
    float4 world_tangent = float4(0, 0, 0, 1);
    {
        const float4 vertex_tangent = draw.CornerTangent.ValuesSlot != InvalidSlot ?
            scene.CornerTangent(draw.CornerTangent, attribute_handle) :
            float4(0, 0, 0, 1);
        float3 tangent = vertex_tangent.xyz;
        if (dot(tangent, tangent) > 1e-8f) {
            tangent = normalize(tangent);
            float3 tangent_dummy_pos = float3(0.0f);
            ApplyArmatureDeform(scene, draw, tangent_dummy_pos, idx, tangent);
            tangent = normalize(trs_transform_normal(world, tangent));
            world_tangent = float4(tangent, vertex_tangent.w);
        }
    }
    const float3 world_scale = float3(world.S);
    return {
        .Position = scene.ViewProj() * float4(world_pos, 1.0f),
        .PointSize = PointSize,
        .WorldNormal = shading_normal ? trs_transform_normal(world, normal) : float3(0.0f),
        .WorldPosition = world_pos,
        .Color = color,
        .FaceOverlayFlags = face_overlay_flags,
        .TexCoord0 = draw.CornerUvs[0].ValuesSlot != InvalidSlot ? scene.CornerUv(draw.CornerUvs[0], attribute_handle) : float2(0),
        .TexCoord1 = draw.CornerUvs[1].ValuesSlot != InvalidSlot ? scene.CornerUv(draw.CornerUvs[1], attribute_handle) : float2(0),
        .TexCoord2 = draw.CornerUvs[2].ValuesSlot != InvalidSlot ? scene.CornerUv(draw.CornerUvs[2], attribute_handle) : float2(0),
        .TexCoord3 = draw.CornerUvs[3].ValuesSlot != InvalidSlot ? scene.CornerUv(draw.CornerUvs[3], attribute_handle) : float2(0),
        .MaterialIndex = material_index,
        .VertexColor = draw.CornerColor.ValuesSlot != InvalidSlot ?
            scene.CornerColor(draw.CornerColor, attribute_handle) :
            float4(1.0f),
        .WorldTangent = world_tangent,
        .WorldScale = face_attributes ? (world_scale.x + world_scale.y + world_scale.z) / 3.0f : 0.0f,
    };
}

#endif
