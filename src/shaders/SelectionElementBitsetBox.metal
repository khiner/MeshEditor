#ifndef SELECTIONELEMENTBITSETBOX_MSL
#define SELECTIONELEMENTBITSETBOX_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "SelectionElementPushConstants.metal"

// Box selection: every element covering a pixel inside the box sets its bit. It writes no color.
[[early_fragment_tests]]
fragment void SelectionElementBitsetBoxFragment(
    ElementIdVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (in.ElementId == 0u) return;
    const uint2 frag_px = uint2(in.Position.xy);
    const uint4 box = uint4(pc.Box);
    if (frag_px.x < box.x || frag_px.x > box.z || frag_px.y < box.y || frag_px.y > box.w) return;
    const uint bit_idx = in.ElementId - 1u;
    device atomic_uint *bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.BoxResultSlot);
    atomic_fetch_or_explicit(&bits[bit_idx >> 5u], 1u << (bit_idx & 31u), memory_order_relaxed);
}

#endif
