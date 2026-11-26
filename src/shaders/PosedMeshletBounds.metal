#pragma once
#include "gpu/AABB.h"
#include "BoundsShared.metal"
#include "MeshletShared.metal"
#include "gpu/PosedMeshletBoundsPushConstants.h"

kernel void PosedMeshletBoundsKernel(
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    SCENE_ARGS,
    constant PosedMeshletBoundsPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    device const PosedMeshletBoundsJob *jobs = BindlessBuffer(PosedMeshletBoundsJob,bindless.Buffer,pc.JobsSlot);
    const auto job = jobs[PartitionPoint(pc.JobCount, [&](uint k) { return jobs[k].FirstGroup <= group_id; }) - 1u];
    const uint instance_id = job.Instance;
    const uint member = Resolve<RenderBlockSize>(SetTable(bindless,pc.Members,job.MemberSet),job.Members+group_id-job.FirstGroup);
    const uint meshlet_address = Resolve<RenderBlockSize>(SetTable(bindless,pc.Meshlets,job.MeshletSet),BindlessBuffer(uint,bindless.Buffer,pc.MemberSlot)[member]);
    if (meshlet_address == InvalidOffset) return;
    const InstanceRecord instance = scene.InstanceRecords(view.InstanceRecordSlot)[instance_id];
    const MeshRecord mesh = scene.MeshRecords(view.MeshRecordSlot)[instance.Mesh];
    const uint destination = PoseAttributeIndex(bindless,pc.PosedMeshletBoundsNodesSlot,PoseNamespace(instance.MeshletBoundsNamespace,mesh.Display.MeshletBoundsNamespace),meshlet_address);
    if (destination == InvalidOffset) return;
    const MeshletRecord meshlet = BindlessBuffer(MeshletRecord,bindless.Buffer,pc.MeshletSlot)[meshlet_address];
    const DrawData draw = ComposeDraw(mesh, instance, instance_id);
    float3 lo = AabbEmptyMin;
    float3 hi = AabbEmptyMax;
    // One 32-lane SIMD group reduces a meshlet, and each lane folds up to two vertices.
    for (uint v = tid; v < meshlet.VertexCount; v += 32u) {
        const uint source_vertex = BindlessBuffer(uint, bindless.Buffer, pc.MeshletVertexSlot)[mesh.Display.Payloads.VertexCorners + meshlet.VertexOffset + v];
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
