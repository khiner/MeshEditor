#ifndef POSED_MESHLET_BOUNDS_MSL
#define POSED_MESHLET_BOUNDS_MSL

// Writes one posed AABB per meshlet by resolving representative corners to canonical vertices.
#include "AABB.metal"
#include "BoundsShared.metal"
#include "MeshletShared.metal"
#include "PosedMeshletBoundsPushConstants.metal"
#include "PrimitiveRecord.metal"

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
    const uint2 tile = uint2(scene.TileMap(pc.TileMapSlot)[group_id]);
    const DrawData entry = scene.Draws(pc.DrawDataSlot)[tile.x];
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord, bindless.Buffer, pc.MeshletSlot)[tile.y];
    const PrimitiveRecord primitive = BindlessBuffer(PrimitiveRecord, bindless.Buffer, pc.PrimitiveSlot)[meshlet.Primitive];
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    if (tid < meshlet.VertexCount) {
        const uint packed_vertex = MeshletPackedVertex(bindless, pc.MeshletVertexSlot, meshlet, tid);
        const uint topology = meshlet.LocalTriangleOffset >> MeshletGeometryEncoding_TopologyShift;
        const uint vertex_id = MeshletVertexId(scene, primitive.Draw, topology, packed_vertex);
        const float3 position = float3(scene.PosedPositions(scene.View.PosedPositionSlot)[entry.PosedPositionOffset + vertex_id]);
        lo = position;
        hi = position;
    }
    FoldSharedAabb(shared_min, shared_max, MeshletBoundsFoldLanes, tid, lo, hi);
    if (tid == 0u) {
        BindlessBufferMutable(AABB, bindless.Buffer, pc.PosedMeshletBoundsSlot)[group_id] = {
            packed_float3(shared_min[0]), packed_float3(shared_max[0])
        };
    }
}

#endif
