#ifndef POSED_MESHLET_BOUNDS_MSL
#define POSED_MESHLET_BOUNDS_MSL

// Writes one posed AABB per meshlet by resolving representative corners to canonical vertices.
#include "gpu/AABB.h"
#include "BoundsShared.metal"
#include "MeshletShared.metal"
#include "gpu/PosedMeshletBoundsPushConstants.h"
#include "ElementWorkShared.metal"
#include "MeshletIndexShared.metal"

kernel void PosedMeshletBoundsKernel(
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant PosedMeshletBoundsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    uint instance_id = pc.Instance, meshlet_id;
    if (pc.Work.Storage.Slot == InvalidSlot) {
        device const PosedMeshletBoundsJob *jobs = BindlessBuffer(PosedMeshletBoundsJob,bindless.Buffer,pc.JobsSlot);
        uint lo = 0u, hi = pc.JobCount;
        while (lo+1u < hi) { const uint mid = (lo+hi)/2u; if (jobs[mid].FirstGroup <= group_id) lo = mid; else hi = mid; }
        const auto job = jobs[lo];
        instance_id = job.Instance;
        meshlet_id = MeshletIndexSelect(bindless,job.Clusters,group_id-job.FirstGroup);
    } else meshlet_id = WorkGroupElement(bindless,pc.Work,group_id);
    if (meshlet_id == InvalidOffset) return;
    const InstanceRecord instance = scene.InstanceRecords(view.InstanceRecordSlot)[instance_id];
    const MeshRecord mesh = scene.MeshRecords(view.MeshRecordSlot)[instance.Mesh];
    const uint destination = PoseAttributeIndex(bindless,pc.PosedMeshletBoundsNodesSlot,PoseNamespace(instance.MeshletBoundsNamespace,mesh.Display.MeshletBoundsNamespace),meshlet_id);
    if (destination == InvalidOffset) return;
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord,bindless.Buffer,pc.MeshletSlot)[meshlet_id];
    const DrawData draw = ComposeDraw(mesh, instance, instance_id);
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    // One 32-lane SIMD group owns a meshlet. Each lane folds up to two vertices.
    for (uint v = tid; v < meshlet.VertexCount; v += 32u) {
        const uint source_vertex = MeshletSourceVertex(bindless, pc.MeshletVertexSlot, meshlet, v);
        const uint topology = meshlet.Topology;
        const uint vertex_id = MeshletVertexId(scene, draw, topology, source_vertex);
        const float3 position = scene.GetLocalPosition(draw, vertex_id);
        lo = min(lo, position);
        hi = max(hi, position);
    }
    lo = simd_min(lo);
    hi = simd_max(hi);
    if (tid == 0u) {
        BindlessBufferMutable(AABB, bindless.Buffer, pc.PosedMeshletBoundsSlot)[destination] = {
            packed_float3(lo), packed_float3(hi)
        };
    }
}

#endif
