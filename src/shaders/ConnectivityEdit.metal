#include "ElementWorkShared.metal"
#include "gpu/ConnectivityEditPushConstants.h"

// The replaced core halfedges and faces leave the neighborhood, and the emitted handles join it.
kernel void ConnectivityEditWork(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant ConnectivityEditPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 tid [[thread_position_in_grid]]
) {
    const uint domain = tid.z, i = tid.x;
    uint handle;
    if (i < pc.Counts[domain]) {
        handle = WorkGroupElement(b,pc.Before[domain],i);
        if (domain != 0u && WorkRank(b,pc.Replaced[domain - 1u],handle) != InvalidOffset) return;
    } else {
        const auto emitted = pc.Emitted[domain];
        const uint index = i-pc.Counts[domain];
        if (index >= emitted.Count) return;
        handle = emitted.Handles.Slot == InvalidSlot ? emitted.First+index :
            BindlessBuffer(uint,b.Buffer,emitted.Handles.Slot)[emitted.Handles.Offset+index];
    }
    if (handle >= pc.After[domain].Count) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,pc.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    MarkWork(b,pc.After[domain],handle);
}
