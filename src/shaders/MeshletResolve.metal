#ifndef MESHLET_RESOLVE_MSL
#define MESHLET_RESOLVE_MSL

#include "gpu/InstanceRecord.h"
#include "gpu/MeshletDrawPushConstants.h"
#include "gpu/MeshletRouteState.h"
#include "MeshletShared.metal"
#include "gpu/PrimitiveRecord.h"
#include "gpu/VisibleMeshlet.h"
#include "TransformUtils.metal"
#include "EditSelection.metal"

struct MeshletWork {
    InstanceRecord Instance;
    MeshletRecord Meshlet;
    PrimitiveRecord Primitive;
    DrawData Draw;
    uint VisibleIndex, MeshletIndex;
    bool Valid;
};

inline MeshletWork ResolveMeshletWork(
    const thread Scene &scene, constant MeshletDrawPushConstants &pc, uint group_index
) {
    device const BindlessSet &bindless = scene.B;
    const MeshletRouteState routes = BindlessBuffer(MeshletRouteState, bindless.Buffer, pc.RouteStateSlot)[0];
    const uint visible_index = routes.Offsets[pc.Route] + pc.VisibleOffset + group_index;
    const VisibleMeshlet work = BindlessBuffer(VisibleMeshlet, bindless.Buffer, pc.VisibleMeshletSlot)[visible_index];
    const uint instance_slot = BindlessBuffer(uint, bindless.Buffer, pc.InstanceMapSlot)[work.Instance];
    const InstanceRecord instance = BindlessBuffer(InstanceRecord, bindless.Buffer, pc.InstanceSlot)[instance_slot];
    const MeshRecord mesh = scene.MeshRecords(scene.View.MeshRecordSlot)[work.Mesh];
    const uint flags = InstanceFlags(mesh.Display.Flags, mesh.Display.PrimaryEditInstanceIndex, instance_slot, uint(scene.InstanceStates(scene.View.InstanceStateSlot)[instance_slot]));
    if ((flags & pc.RequiredInstanceFlags) != pc.RequiredInstanceFlags) {
        return {.Instance = instance, .VisibleIndex = visible_index, .MeshletIndex = work.Meshlet};
    }
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[work.Meshlet];
    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    return {
        .Instance = instance,
        .Meshlet = meshlet,
        .Primitive = primitive,
        .Draw = ComposeDraw(mesh, instance, instance_slot, meshlet.Topology),
        .VisibleIndex = visible_index,
        .MeshletIndex = work.Meshlet,
        .Valid = true,
    };
}

struct MeshletFaceValues {
    float3 FlatWorldNormal;
    uint FaceOverlayFlags, MaterialIndex;
    float WorldScale;
    uint ObjectId, ElementId;
};

// Returns true for clusters with independent triangles and no source-triangle or source-face identity.
inline bool MeshletCoarse(MeshletRecord meshlet) { return meshlet.RefinedGroup != InvalidOffset; }

// Finest element visibility is independent of cluster ownership and geometry.
inline bool MeshletElementHidden(const thread Scene &scene,const thread MeshletWork &work,
    constant MeshletDrawPushConstants &pc,uint element) {
    if (scene.View.InteractionMode!=InteractionMode::Edit || MeshletCoarse(work.Meshlet)) return false;
    const uint handle=BindlessBuffer(uint,scene.B.Buffer,pc.MeshletTriangleSlot)[work.Meshlet.TriangleOffset+element];
    const uint topology=MeshletPrimitiveTopology(work.Meshlet);
    if (topology==uint(MeshPrimitiveTopology::Triangle)) return EditElementHidden(scene,work.Draw,Element::Face,scene.TriangleFace(work.Draw,handle));
    const bool line=topology==uint(MeshPrimitiveTopology::Line);
    return EditElementHidden(scene,work.Draw,line ? Element::Edge : Element::Vertex,
        handle+(line ? work.Draw.Connectivity.Edges.Offset : work.Draw.VertexOffset));
}

// Returns attribute corners from the cluster vertex list or original source triangle.
inline uint3 MeshletCornerIds(
    device const BindlessSet &bindless, uint vertex_slot, uint local_triangle_slot,
    MeshletRecord meshlet, DrawData draw, uint triangle, uint local_triangle
) {
    if (!MeshletCoarse(meshlet)) {
        if (draw.TriangleSlot != InvalidSlot) return uint3(BindlessBuffer(packed_uint3, bindless.Buffer, draw.TriangleSlot)[triangle]);
        const uint base = draw.IndexSlotOffset.Offset + triangle * 3u;
        return uint3(base, base + 1u, base + 2u);
    }
    device const uchar *triangles = BindlessBuffer(uchar, bindless.Buffer, local_triangle_slot);
    device const uint *vertices = BindlessBuffer(uint, bindless.Buffer, vertex_slot);
    const uint offset = MeshletLocalTriangleOffset(meshlet) + local_triangle * 3u;
    uint3 corners;
    for (uint c = 0u; c < 3u; ++c) {
        const uint local = uint(triangles[offset + c] & uint(MeshletGeometryEncoding::LocalIndexMask));
        corners[c] = vertices[meshlet.VertexOffset + local];
    }
    return corners;
}

// Returns a coarse triangle's mesh-local face normal under source-geometry winding.
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
    MeshletRecord meshlet, uint triangle, uint local_triangle
) {
    const uint3 corner_ids = MeshletCornerIds(
        scene.B, vertex_slot, local_triangle_slot, meshlet, draw, triangle, local_triangle
    );
    const uint3 vertex_ids{
        scene.CornerVertexOrdinal(draw, corner_ids.x),
        scene.CornerVertexOrdinal(draw, corner_ids.y),
        scene.CornerVertexOrdinal(draw, corner_ids.z),
    };
    return {.CornerIds = corner_ids, .VertexIds = vertex_ids, .CoarseNormal = MeshletCoarse(meshlet) ? MeshletCoarseNormal(scene, draw, vertex_ids) : float3(0.0f)};
}

// Returns coarse face values with primitive material and no source-face selection state.
inline MeshletFaceValues MeshletCoarseFace(
    const thread Scene &scene, PrimitiveRecord primitive, InstanceRecord instance, DrawData draw, Transform world
) {
    const float3 scale = float3(world.S);
    return {
        float3(0.0f),
        0u,
        MeshletPrimitiveMaterialIndex(scene, primitive),
        (scale.x + scale.y + scale.z) / 3.0f,
        instance.ObjectId,
        draw.ElementIdOffset,
    };
}

inline uint MeshletFaceMaterialIndex(
    const thread Scene &scene, DrawData draw, uint face_id
) {
    if (draw.ElementPrimitives.ValuesSlot == InvalidSlot || draw.PrimitiveMaterialOffset == InvalidOffset || face_id == InvalidOffset) return 0u;
    const uint primitive_index = scene.ElementPrimitive(draw.ElementPrimitives, face_id);
    return scene.PrimitiveMaterials(scene.View.PrimitiveMaterialSlot)[draw.PrimitiveMaterialOffset + primitive_index];
}

inline MeshletFaceValues MeshletFace(
    const thread Scene &scene, DrawData draw, InstanceRecord instance,
    Transform world, uint triangle, bool flat_face
) {
    const uint face_id = scene.TriangleFace(draw, triangle);
    const uint element_state = scene.View.InteractionMode == InteractionMode::Edit && face_id != InvalidOffset ?
        EditFaceState(scene, draw, face_id) : 0u;
    const uint material_index = MeshletFaceMaterialIndex(scene, draw, face_id);
    float3 flat_world_normal = float3(0.0f);
    if (flat_face) flat_world_normal = trs_transform_normal(world, scene.GetFaceNormal(draw, face_id));
    const float3 scale = float3(world.S);
    return {
        flat_world_normal,
        ((element_state & STATE_SELECTED) != 0u ? 1u : 0u) |
            ((element_state & STATE_ACTIVE) != 0u ? 2u : 0u) | (flat_face ? 4u : 0u),
        material_index,
        (scale.x + scale.y + scale.z) / 3.0f,
        instance.ObjectId,
        scene.FacePickId(draw, face_id),
    };
}

#endif
