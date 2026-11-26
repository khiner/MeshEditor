#pragma once
#include "gpu/GeometryEditJob.h"
#include "ElementWorkShared.metal"
#include "ConnectivityRead.metal"

kernel void FinalizeElementWorkKernel(
    uint tid [[thread_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant ElementWork *work [[buffer(BufferIndex_PushConstants)]]
) {
    threadgroup uint totals[8];
    FinishWork(bindless, work[group], tid, totals);
}

inline void MarkOwnedMeshlet(device const BindlessSet &b, constant GeometryEditJob &job, device const uint *meshlets, uint topology, uint handle) {
    const auto map=job.ElementMeshlets[topology];
    if (map.Slot == InvalidSlot) return;
    const uint owner = BindlessBuffer(uint,b.Buffer,map.Slot)[map.Offset+handle];
    if (owner != InvalidOffset) MarkWork(b,job.MeshletAddresses,Resolve<RenderBlockSize>(meshlets,owner));
}

// Each job's tile k runs the elements of the k-th occupied block of the work its phase reads, and tiles past the work's blocks do nothing.
kernel void GeometryEditKernel(JOB_TABLE_ARGS(GeometryEditJob)) {
    JOB_THREAD(job, invocation);
    if (table.Parameter == 4u) {
        const uint i = WorkElement(b, job.Candidates, invocation);
        if (i == InvalidOffset) return;
        if (job.Mode == GeometryEditMode::Commit) return;
        MarkWork(b, job.ChangedVertices, i);
    } else if (table.Parameter == 1u) {
        const uint i = WorkElement(b, job.Candidates, invocation);
        if (i == InvalidOffset) return;
        const ConnectivityView conn{b, job.Connectivity};
        const auto fan=conn.Fan(i);
        device const uint *meshlets = SetTable(b, job.MeshletSets, job.MeshletSet);
        if (!fan.Count) MarkOwnedMeshlet(b,job,meshlets,2u,i);
        for (const auto item : fan) {
            if (item.y==InvalidOffset) MarkOwnedMeshlet(b,job,meshlets,1u,conn.Edge(item.x));
            else MarkWork(b, job.Faces, item.y);
        }
    } else {
        const uint f = WorkElement(b, job.Faces, invocation);
        if (f == InvalidOffset) return;
        device const uint *triangles = BindlessBuffer(uint, b.ObjectIdBuffer, job.FaceTriangleStartSlot);
        const ConnectivityView conn{b, job.Connectivity};
        const uint2 loop = conn.FaceHalfedges(f);
        const uint first = triangles[conn.FaceAddress(f)], end = first + loop.y - loop.x - 2u;
        device const uint *meshlets = SetTable(b, job.MeshletSets, job.MeshletSet);
        for (uint t = first; t < end; ++t) MarkOwnedMeshlet(b,job,meshlets,0u,t);
        for (uint h = loop.x; h < loop.y; ++h) MarkWork(b, job.Normals, conn.VertexOf(h));
    }
}
