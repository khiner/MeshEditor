#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/ElementRepairJob.h"

// A live element that starts drawing without an owner is added, and an owned element that stopped drawing is removed.
kernel void ElementRepairMark(JOB_TABLE_ARGS(ElementRepairJob)) {
    JOB_TILE_THREAD(job, slot, MeshElementBlockSize);
    uint handle = WorkElement(b, job.Work, slot);
    if (handle == InvalidOffset) return;
    const ConnectivityView view{b, job.Connectivity};
    const uint d = job.WorkDomain == 0u ? 0u : 1u;
    if (job.WorkDomain == 2u) handle = view.Edge(handle);
    if (handle >= max(job.Bound[d], job.OwnerBound[d])) return;
    const uint entry = handle < job.Bound[d] ? (d == 0u ? view.VertexTable : view.EdgeTable)[handle / MeshElementBlockSize] : InvalidOffset;
    bool live = false;
    if (entry != InvalidOffset) {
        const uint slot = handle % MeshElementBlockSize;
        live = SlotLive(BindlessBuffer(MeshElementBlock, b.Buffer, job.MembershipSlot[d])[entry], slot);
    }
    const bool draws = live && (d == 0u ? view.Incoming(handle).y == 0u : view.HalfedgeFace(view.EdgeHalfedge(handle)) == InvalidOffset);
    uint owner = InvalidOffset;
    if (job.Owners[d].Slot != InvalidSlot && handle < job.OwnerBound[d]) owner = BindlessBuffer(uint, b.Buffer, job.Owners[d].Slot)[job.Owners[d].Offset + handle];
    device uint *bits = nullptr;
    if (draws && owner == InvalidOffset) bits = ScratchOf(b, table).Run(job.Added[d]);
    else if (!draws && owner != InvalidOffset) bits = ScratchOf(b, table).Run(job.Removed[d]);
    if (bits) atomic_fetch_or_explicit(AtomicWord(bits + handle / 32u), 1u << (handle % 32u), memory_order_relaxed);
}
