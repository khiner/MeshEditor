#include "Bindless.metal"
#include "gpu/MeshletOwnersPushConstants.h"
#include "gpu/MeshletRecord.h"

// Writes the owner entry of each element a published cluster's payload names.
// The record's owner range covers every element handle.
kernel void MeshletOwners(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshletOwnersPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    if (group >= pc.Count) return;
    const uint cluster = pc.First+group;
    const auto record = BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[Resolve<RenderBlockSize>(SetTable(b,pc.Meshlets,pc.MeshletSet),cluster)];
    if (lane >= record.TriangleCount) return;
    const uint element = BindlessBuffer(uint,b.Buffer,pc.TriangleIdsSlot)[pc.TriangleIdBase+record.TriangleOffset+lane];
    if (record.Topology != pc.Topology || record.RefinedGroup != InvalidOffset || element >= pc.HandleBound) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    BindlessBufferMutable(uint,b.Buffer,pc.Owners.Slot)[pc.Owners.Offset+element] = cluster;
}
