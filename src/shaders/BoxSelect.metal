#ifndef BOXSELECT_MSL
#define BOXSELECT_MSL

#include "SelectionTraversal.metal"
#include "BoxSelectPushConstants.metal"

kernel void BoxSelectKernel(
    uint2 local [[thread_position_in_grid]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant BoxSelectPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint2 box_min = uint2(pc.BoxMin), box_max = uint2(pc.BoxMax);
    const uint2 pixel = box_min + local;
    if (pixel.x > box_max.x || pixel.y > box_max.y) return;

    // One bit per id, packed into 32-bit words: `>> 5` selects the word, `& 31` the bit.
    device atomic_uint *bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.BoxResultIndex);
    const uint2 extent = uint2(pc.HeadExtent);

    uint idx = SelectionHeadAt(scene, pc.HeadSlot, extent, int2(pixel));
    while (idx != INVALID_SELECTION_NODE) {
        const SelectionNode node = SelectionNodeAt(scene, pc.SelectionNodesIndex, idx);
        if (SelectionIdInRange(node.Id, pc.MaxId)) {
            const uint bit_idx = node.Id - 1u;
            atomic_fetch_or_explicit(&bits[bit_idx >> 5u], 1u << (bit_idx & 31u), memory_order_relaxed);
        }
        idx = node.Next;
    }
}

#endif
