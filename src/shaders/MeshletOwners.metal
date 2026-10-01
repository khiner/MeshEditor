#include "Bindless.metal"
#include "gpu/MeshletOwnersPushConstants.h"
#include "gpu/MeshletRecord.h"

// Writes the owner entry of each element a published cluster's payload names.
// The host attached the payload block of every such element.
kernel void MeshletOwners(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant MeshletOwnersPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    if (group >= pc.Count) return;
    const uint cluster = pc.First+group;
    const auto record = BindlessBuffer(MeshletRecord,b.Buffer,pc.MeshletSlot)[cluster];
    if (lane >= record.TriangleCount) return;
    const uint element = pc.ElementOrigin+BindlessBuffer(uint,b.Buffer,pc.TriangleIdsSlot)[record.TriangleOffset+lane];
    const uint payload = element/256u < pc.BlockCount ? BindlessBuffer(uint,b.Buffer,pc.Owners.BlocksSlot)[element/256u] : 0u;
    if (record.Topology != pc.Topology || record.RefinedGroup != InvalidOffset || !payload) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    BindlessBufferMutable(uint,b.Buffer,pc.Owners.ValuesSlot)[(payload-1u)*256u+element%256u] = cluster;
}
