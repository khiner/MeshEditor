#ifndef OBJECTPICK_MSL
#define OBJECTPICK_MSL

#include "SelectionTraversal.metal"
#include "ObjectPickPushConstants.metal"

constant uint ObjectPickGroupSize = 256;

// The pick key orders candidates by radial distance first, then depth, within this click's epoch.
inline uint PackKey(uint epoch_inv, uint dist_sq, float depth) {
    const uint dist_u8 = min(dist_sq, 0xffu);
    const uint depth_u16 = uint(clamp(depth, 0.0f, 1.0f) * 65535.0f + 0.5f);
    return (epoch_inv << 24u) | (dist_u8 << 16u) | depth_u16;
}

kernel void ObjectPickKernel(
    uint local [[thread_position_in_threadgroup]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ObjectPickPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};
    const uint max_id = pc.MaxId;
    if (max_id == 0u) return;

    device atomic_uint *best_keys = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.BestKeyIndex);
    device atomic_uint *seen_bits = BindlessBufferMutable(atomic_uint, bindless.Buffer, pc.SeenBitsIndex);

    // Clear this click's seen-object bitset.
    const uint bit_words = (max_id + 31u) / 32u;
    for (uint i = local; i < bit_words; i += ObjectPickGroupSize) {
        atomic_store_explicit(&seen_bits[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    const uint diameter = pc.Radius * 2u + 1u;
    const uint pixel_count = diameter * diameter;
    const uint2 extent = uint2(pc.HeadExtent);
    const int2 size = int2(extent);

    for (uint i = local; i < pixel_count; i += ObjectPickGroupSize) {
        const int dx = int(i % diameter) - int(pc.Radius);
        const int dy = int(i / diameter) - int(pc.Radius);
        const uint dist_sq = uint(dx * dx + dy * dy);
        if (dist_sq > pc.Radius * pc.Radius) continue;
        const int2 pixel = int2(uint2(pc.TargetPx)) + int2(dx, dy);
        if (!SelectionPixelInBounds(pixel, size)) continue;

        uint node_idx = SelectionHeadAt(scene, pc.HeadSlot, extent, pixel);
        while (node_idx != INVALID_SELECTION_NODE) {
            const SelectionNode node = SelectionNodeAt(scene, pc.SelectionNodesIndex, node_idx);
            if (SelectionIdInRange(node.Id, max_id)) {
                const uint idx = node.Id - 1u;
                atomic_fetch_min_explicit(&best_keys[idx], PackKey(pc.EpochInv, dist_sq, node.Depth), memory_order_relaxed);
                atomic_fetch_or_explicit(&seen_bits[idx >> 5u], 1u << (idx & 31u), memory_order_relaxed);
            }
            node_idx = node.Next;
        }
    }
}

#endif
