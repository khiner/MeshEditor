#ifndef POSED_MESHLET_BOUNDS_MSL
#define POSED_MESHLET_BOUNDS_MSL

// Writes one posed AABB per meshlet by resolving representative corners to canonical vertices.
#include "gpu/AABB.h"
#include "BoundsShared.metal"
#include "MeshletShared.metal"
#include "gpu/PosedMeshletBoundsPushConstants.h"
#include "gpu/PrimitiveRecord.h"
#include "ElementWorkShared.metal"

kernel void PosedMeshletBoundsKernel(
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float3 *shared_min [[threadgroup(0)]],
    threadgroup float3 *shared_max [[threadgroup(1)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant PosedMeshletBoundsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint work_id = pc.Work.Storage.Slot == InvalidSlot ? group_id : WorkElement(bindless, pc.Work, group_id);
    if (work_id == InvalidOffset) return;
    const uint destination = pc.Work.Storage.Slot == InvalidSlot ? group_id : pc.FirstTile + work_id;
    const uint2 tile = uint2(scene.TileMap(pc.TileMapSlot)[destination]);
    const InstanceRecord instance = scene.InstanceRecords(view.InstanceRecordSlot)[scene.BoundsEntries(pc.BoundsEntrySlot)[tile.x].FirstInstance];
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[tile.y];
    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    const MeshRecord mesh = scene.MeshRecords(view.MeshRecordSlot)[instance.Mesh];
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    if (tid < meshlet.VertexCount) {
        const uint packed_vertex = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, meshlet, tid);
        const uint topology = meshlet.LocalTriangleOffset >> uint(MeshletGeometryEncoding::TopologyShift);
        const uint vertex_id = MeshletVertexId(scene, ComposeDraw(mesh, primitive.FirstTriangle, instance, 0u, EditSelectionStorage{}), topology, packed_vertex);
        const float3 position = float3(scene.PosedPositions(scene.View.PosedPositionSlot)[instance.PosedPositionOffset + vertex_id]);
        lo = position;
        hi = position;
    }
    FoldSharedAabb(shared_min, shared_max, MeshletBoundsFoldLanes, tid, lo, hi);
    if (tid == 0u) {
        BindlessBufferMutable(AABB, bindless.Buffer, pc.PosedMeshletBoundsSlot)[destination] = {
            packed_float3(shared_min[0]), packed_float3(shared_max[0])
        };
    }
}

#endif
