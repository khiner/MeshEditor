#include "ConnectivityRead.metal"
#include "ElementWorkShared.metal"
#include "gpu/ElementRepairPushConstants.h"

// A live element that starts drawing without an owner is added, and an owned element that stopped drawing is removed.
kernel void ElementRepairMark(uint i [[thread_position_in_grid]], device const BindlessSet &b [[buffer(BufferIndex_Bindless)]], constant ElementRepairPushConstants &pc [[buffer(BufferIndex_PushConstants)]]) {
    if (i >= pc.WorkCount) return;
    uint handle = WorkElement(b, pc.Work, i);
    if (handle == InvalidOffset) return;
    const ConnectivityView view{b, pc.Connectivity};
    const uint d = pc.WorkDomain == 0u ? 0u : 1u;
    if (pc.WorkDomain == 2u) handle = view.Edge(handle);
    if (handle >= max(pc.Bound[d], pc.OwnerBound[d])) return;
    const uint entry = handle < pc.Bound[d] ? (d == 0u ? view.VertexTable : view.EdgeTable)[handle / MeshElementBlockSize] : InvalidOffset;
    bool live = false;
    if (entry != InvalidOffset) {
        const uint slot = handle % MeshElementBlockSize;
        live = SlotLive(BindlessBuffer(MeshElementBlock, b.Buffer, pc.MembershipSlot[d])[entry], slot);
    }
    const bool draws = live && (d == 0u ? view.Incoming(handle).y == 0u : view.HalfedgeFace(view.EdgeHalfedge(handle)) == InvalidOffset);
    uint owner = InvalidOffset;
    if (pc.Owners[d].Slot != InvalidSlot && handle < pc.OwnerBound[d]) owner = BindlessBuffer(uint, b.Buffer, pc.Owners[d].Slot)[pc.Owners[d].Offset + handle];
    device uint *bits = nullptr;
    if (draws && owner == InvalidOffset) bits = BindlessBufferMutable(uint, b.Buffer, pc.ScratchSlot) + pc.Added[d];
    else if (!draws && owner != InvalidOffset) bits = BindlessBufferMutable(uint, b.Buffer, pc.ScratchSlot) + pc.Removed[d];
    if (bits) atomic_fetch_or_explicit(AtomicWord(bits + handle / 32u), 1u << (handle % 32u), memory_order_relaxed);
}
