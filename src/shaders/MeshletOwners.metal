#include "Bindless.metal"
#include "gpu/MeshletOwnersJob.h"
#include "gpu/MeshletRecord.h"

// Writes the owner entry of each element a published cluster's payload names, with one threadgroup per cluster.
// The record's owner range covers every element handle.
kernel void MeshletOwners(JOB_TABLE_ARGS(MeshletOwnersJob)) {
    JOB_TILE(job);
    const uint cluster = job.First+tile;
    const auto record = BindlessBuffer(MeshletRecord,b.Buffer,job.MeshletSlot)[Resolve<RenderBlockSize>(SetTable(b,job.Meshlets,job.MeshletSet),cluster)];
    if (lane >= record.TriangleCount) return;
    const uint element = BindlessBuffer(uint,b.Buffer,job.TriangleIdsSlot)[job.TriangleIdBase+record.TriangleOffset+lane];
    if (record.Topology != job.Topology || record.RefinedGroup != InvalidOffset || element >= job.HandleBound) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,job.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    BindlessBufferMutable(uint,b.Buffer,job.Owners.Slot)[job.Owners.Offset+element] = cluster;
}
