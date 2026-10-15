#include "Bindless.metal"
#include "gpu/MeshletOwnersPushConstants.h"
#include "gpu/MeshletRecord.h"

// Writes the owner entry of each element a published cluster's payload names, with one threadgroup per cluster.
// The record's owner range covers every element handle.
kernel void MeshletOwners(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshletOwnersPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    if (group >= pc.Count) return;
    device const uint *tile = BindlessBuffer(uint,b.Buffer,pc.Tiles.Slot)+pc.Tiles.Offset+2u*group;
    const MeshletOwnersJob job = reinterpret_cast<device const MeshletOwnersJob *>(BindlessBuffer(uint,b.Buffer,pc.Jobs.Slot)+pc.Jobs.Offset)[tile[0]];
    const uint cluster = job.First+tile[1];
    const auto record = BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[Resolve<RenderBlockSize>(SetTable(b,pc.Meshlets,job.MeshletSet),cluster)];
    if (lane >= record.TriangleCount) return;
    const uint element = BindlessBuffer(uint,b.Buffer,pc.TriangleIdsSlot)[job.TriangleIdBase+record.TriangleOffset+lane];
    if (record.Topology != job.Topology || record.RefinedGroup != InvalidOffset || element >= job.HandleBound) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    BindlessBufferMutable(uint,b.Buffer,job.Owners.Slot)[job.Owners.Offset+element] = cluster;
}
