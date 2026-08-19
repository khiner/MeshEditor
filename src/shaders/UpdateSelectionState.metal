#ifndef UPDATESELECTIONSTATE_MSL
#define UPDATESELECTIONSTATE_MSL

// Reads the selection bitset and writes element state bytes for one mesh.
// Dispatched after the fragment selection pass, or after a direct bit write for click-select.
// One thread per element (vertex, edge, or face).
// Edge mode: the state buffer holds 2 bytes per edge, one per halfedge direction.
#include "Bindless.metal"
#include "UpdateSelectionStatePushConstants.metal"

constant uint INVALID_HANDLE = 0xffffffffu;

kernel void UpdateSelectionStateKernel(
    uint i [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant UpdateSelectionStatePushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (i >= pc.ElementCount) return;

    const uint global_bit = pc.BitsetOffset + i;
    const uint word = BindlessBuffer(uint, bindless.Buffer, pc.BitsetSlot)[global_bit >> 5u];
    const bool is_selected = ((word >> (global_bit & 31u)) & 1u) != 0u;
    const bool is_active = pc.ActiveHandle != INVALID_HANDLE && i == pc.ActiveHandle;

    uchar state = 0;
    if (is_selected) state |= uchar(STATE_SELECTED);
    if (is_active) state |= uchar(STATE_ACTIVE);

    device uchar *states = BindlessBufferMutable(uchar, bindless.Buffer, pc.StateSlot);
    if (pc.EdgeMode != 0u) {
        states[pc.StateOffset + 2u * i] = state;
        states[pc.StateOffset + 2u * i + 1u] = state;
    } else {
        states[pc.StateOffset + i] = state;
    }
}

#endif
