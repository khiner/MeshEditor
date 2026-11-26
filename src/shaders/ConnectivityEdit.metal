#include "ElementWorkShared.metal"
#include "gpu/ConnectivityEditJob.h"

// Replace removed core halfedges and faces with emitted handles in the neighborhood.
kernel void ConnectivityEditWork(JOB_TABLE_LAYERED_ARGS(ConnectivityEditJob)) {
    JOB_THREAD(job, i);
    const uint domain = group.z;
    uint handle;
    if (i < job.Counts[domain]) {
        handle = WorkGroupElement(b,job.Before[domain],i);
        if (domain != 0u && WorkRank(b,job.Replaced[domain - 1u],handle) != InvalidOffset) return;
    } else {
        const auto emitted = job.Emitted[domain];
        const uint index = i-job.Counts[domain];
        if (index >= emitted.Count) return;
        handle = emitted.Handles.Slot == InvalidSlot ? emitted.First+index :
            BindlessBuffer(uint,b.Buffer,emitted.Handles.Slot)[emitted.Handles.Offset+index];
    }
    if (handle >= job.After[domain].Count) {
        atomic_store_explicit(BindlessBufferMutable(atomic_uint,b.Buffer,job.ErrorSlot),1u,memory_order_relaxed);
        return;
    }
    MarkWork(b,job.After[domain],handle);
}
