#ifndef MESHLET_RESOLVE_MSL
#define MESHLET_RESOLVE_MSL

#include "InstanceRecord.metal"
#include "MeshletDrawPushConstants.metal"
#include "MeshletRouteState.metal"
#include "MeshletShared.metal"
#include "PrimitiveRecord.metal"
#include "VisibleMeshlet.metal"
#include "TransformUtils.metal"

inline DrawData MeshletDraw(PrimitiveRecord primitive, InstanceRecord instance, uint instance_slot) {
    DrawData draw = primitive.Draw;
    draw.FirstInstance = instance_slot;
    draw.BoneDeformOffset = instance.BoneDeformOffset;
    draw.ArmatureDeformOffset = instance.ArmatureDeformOffset;
    draw.MorphDeformOffset = instance.MorphDeformOffset;
    draw.MorphWeightsOffset = instance.MorphWeightsOffset;
    draw.MorphTargetCount = instance.MorphTargetCount;
    draw.PosedPositionOffset = instance.PosedPositionOffset;
    draw.PosedVertexNormalOffset = instance.PosedVertexNormalOffset;
    draw.PosedSeamNormalOffset = instance.PosedSeamNormalOffset;
    draw.PosedFaceNormalOffset = instance.PosedFaceNormalOffset;
    draw.ElementStateSlotOffset = instance.ElementStateSlotOffset;
    draw.HasPendingVertexTransform = instance.HasPendingVertexTransform;
    draw.PrimaryEditInstanceIndex = instance.PrimaryEditInstanceIndex;
    return draw;
}

struct MeshletWork {
    InstanceRecord Instance;
    MeshletRecord Meshlet;
    PrimitiveRecord Primitive;
    DrawData Draw;
    uint VisibleIndex;
    bool Valid;
};

inline MeshletWork ResolveMeshletWork(
    device const BindlessSet &bindless, constant MeshletDrawPushConstants &pc, uint group_index
) {
    const MeshletRouteState routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot)[0];
    const uint visible_index = routes.Offsets[pc.Route] + pc.VisibleOffset + group_index;
    const VisibleMeshlet work = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot)[visible_index];
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
    MeshletWork result{};
    result.VisibleIndex = visible_index;
    result.Instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
    if ((result.Instance.Flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) return result;
    result.Meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[work.Meshlet];
    result.Primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[result.Meshlet.Primitive];
    result.Draw = MeshletDraw(result.Primitive, result.Instance, instance_slot);
    result.Valid = true;
    return result;
}

struct MeshletFaceValues {
    float3 FlatWorldNormal;
    uint FaceOverlayFlags, MaterialIndex;
    float WorldScale;
    uint ObjectId, ElementId;
};

inline uint MeshletPrimitiveMaterialIndex(
    const thread Scene &scene, PrimitiveRecord primitive
) {
    if (primitive.Draw.PrimitiveMaterialOffset == INVALID_OFFSET) return 0u;
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[
        primitive.Draw.PrimitiveMaterialOffset + primitive.PrimitiveIndex
    ];
}

inline uint MeshletPrimitiveTopology(MeshletRecord meshlet) {
    return meshlet.LocalTriangleOffset >> MeshletGeometryEncoding_TopologyShift;
}

// A cluster simplified from finer geometry. Its triangles are its own, so it names no source triangle
// and no source face.
inline bool MeshletCoarse(MeshletRecord meshlet) { return meshlet.RefinedGroup != INVALID_OFFSET; }

// The three primitive-local corners a rendered triangle reads its attributes from. A coarse cluster
// names them through its own vertex list, and original geometry through its source triangle.
inline uint3 MeshletCornerIds(
    device const BindlessSet &bindless, uint vertex_slot, uint local_triangle_slot,
    MeshletRecord meshlet, PrimitiveRecord primitive, uint triangle, uint local_triangle
) {
    if (!MeshletCoarse(meshlet)) {
        const uint base = (triangle - primitive.FirstTriangle) * 3u;
        return uint3(base, base + 1u, base + 2u);
    }
    device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, local_triangle_slot);
    device const uint *vertices = BindlessBuffer(uint, bindless.Buffer, vertex_slot);
    const uint offset = MeshletLocalTriangleOffset(meshlet) + local_triangle * 3u;
    uint3 corners;
    for (uint c = 0u; c < 3u; ++c) {
        const uint local = uint(triangles[offset + c] & MeshletGeometryEncoding_LocalIndexMask);
        corners[c] = vertices[meshlet.VertexOffset + local] & MeshletGeometryEncoding_CornerMask;
    }
    return corners;
}

// The face normal a coarse triangle's Face-class corners shade from, in mesh-local space and wound
// like the source geometry it replaces.
inline float3 MeshletCoarseNormal(const thread Scene &scene, DrawData draw, uint3 vertex_ids) {
    const float3 p0 = scene.GetLocalPosition(draw, vertex_ids.x);
    return NormalizeOrZero(cross(
        scene.GetLocalPosition(draw, vertex_ids.y) - p0, scene.GetLocalPosition(draw, vertex_ids.z) - p0
    ));
}

struct MeshletTriangleCorners {
    uint3 CornerIds;
    uint3 VertexIds;
    float3 CoarseNormal;
};

inline MeshletTriangleCorners ResolveMeshletCorners(
    const thread Scene &scene, DrawData draw, uint vertex_slot, uint local_triangle_slot,
    MeshletRecord meshlet, PrimitiveRecord primitive, uint triangle, uint local_triangle
) {
    MeshletTriangleCorners result;
    result.CornerIds = MeshletCornerIds(
        scene.B, vertex_slot, local_triangle_slot, meshlet, primitive, triangle, local_triangle
    );
    device const uint *indices = scene.Indices(draw.IndexSlotOffset.Slot);
    for (uint c = 0u; c < 3u; ++c) result.VertexIds[c] = indices[draw.IndexSlotOffset.Offset + result.CornerIds[c]];
    result.CoarseNormal = MeshletCoarse(meshlet) ? MeshletCoarseNormal(scene, draw, result.VertexIds) : float3(0.0f);
    return result;
}

// A coarse cluster's face values: no source face, so no element state and no face normal, and the
// material its primitive carries.
inline MeshletFaceValues MeshletCoarseFace(
    const thread Scene &scene, PrimitiveRecord primitive, InstanceRecord instance, Transform world
) {
    const float3 scale = float3(world.S);
    return {
        float3(0.0f),
        0u,
        MeshletPrimitiveMaterialIndex(scene, primitive),
        (scale.x + scale.y + scale.z) / 3.0f,
        instance.ObjectId,
        instance.ElementIdOffset,
    };
}

inline uint MeshletFaceMaterialIndex(
    const thread Scene &scene, DrawData draw, uint face_id
) {
    if (draw.ElementPrimitiveOffset == INVALID_OFFSET || draw.PrimitiveMaterialOffset == INVALID_OFFSET || face_id == 0u) return 0u;
    const uint primitive_index = scene.ElementPrimitives(scene.View.ElementPrimitiveSlot)[draw.ElementPrimitiveOffset + face_id - 1u];
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[draw.PrimitiveMaterialOffset + primitive_index];
}

inline MeshletFaceValues MeshletFace(
    const thread Scene &scene, DrawData draw, PrimitiveRecord primitive, InstanceRecord instance,
    Transform world, uint triangle, bool flat_face
) {
    const uint face_id = scene.ObjectIds(draw.ObjectIdSlot)[draw.FaceIdOffset + triangle - primitive.FirstTriangle];
    const uint element_state = draw.ElementStateSlotOffset.Slot != INVALID_SLOT && face_id != 0u ?
        uint(scene.ElementStates(draw.ElementStateSlotOffset.Slot)[draw.ElementStateSlotOffset.Offset + face_id - 1u]) : 0u;
    const uint material_index = MeshletFaceMaterialIndex(scene, draw, face_id);
    float3 flat_world_normal = float3(0.0f);
    if (flat_face) flat_world_normal = trs_transform_normal(world, scene.GetFaceNormal(draw, face_id - 1u));
    const float3 scale = float3(world.S);
    return {
        flat_world_normal,
        ((element_state & STATE_SELECTED) != 0u ? 1u : 0u) |
            ((element_state & STATE_ACTIVE) != 0u ? 2u : 0u) | (flat_face ? 4u : 0u),
        material_index,
        (scale.x + scale.y + scale.z) / 3.0f,
        instance.ObjectId,
        instance.ElementIdOffset + face_id,
    };
}

#endif
