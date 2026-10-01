#ifndef SELECTIONELEMENTBITSETBOX_MSL
#define SELECTIONELEMENTBITSETBOX_MSL

#include "Bindless.metal"
#include "Varyings.metal"
#include "gpu/SelectionElementPushConstants.h"

// Sets the bit for every element covering a pixel inside the selection box.
[[early_fragment_tests]]
fragment void SelectionElementBitsetBoxFragment(
    ElementIdFragmentVaryings in [[stage_in]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SelectionElementPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    if (in.ElementId == 0u) return;
    const uint2 frag_px = uint2(in.Position.xy);
    const uint4 box = uint4(pc.Query.Box);
    if (frag_px.x < box.x || frag_px.x > box.z || frag_px.y < box.y || frag_px.y > box.w) return;
    const uint bit_idx = in.ElementId - 1u;
    device atomic_uint *bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.Query.Results.MasksSlot);
    if (atomic_fetch_or_explicit(&bits[bit_idx >> 5u], 1u << (bit_idx & 31u), memory_order_relaxed) == 0u) {
        device atomic_uint *count = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.Query.Results.CountSlot);
        const uint index = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        BindlessBufferMutable(uint, bindless.Buffer, pc.Query.Results.WordsSlot)[index] = bit_idx >> 5u;
    }
}

#endif
