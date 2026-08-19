#ifndef ELEMENTPICK_MSL
#define ELEMENTPICK_MSL

#include "SelectionTraversal.metal"
#include "ElementPickCandidate.metal"
#include "ElementPickPushConstants.metal"

constant uint ElementPickGroupSize = 256;

inline bool IsBetter(ElementPickCandidate a, ElementPickCandidate b) {
    if (a.Id == 0u) return false;
    if (b.Id == 0u) return true;
    if (a.DistanceSq < b.DistanceSq) return true;
    if (a.DistanceSq > b.DistanceSq) return false;
    return a.Depth < b.Depth;
}

kernel void ElementPickKernel(
    uint idx [[thread_position_in_grid]],
    uint local [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup ElementPickCandidate *shared_candidates [[threadgroup(0)]],
    device const BindlessSet &bindless [[buffer(BufferIndex_Bindless)]],
    constant SceneViewUBO &view [[buffer(BufferIndex_SceneView)]],
    constant ViewportTheme &theme [[buffer(BufferIndex_ViewportTheme)]],
    constant WorkspaceLights &workspace [[buffer(BufferIndex_WorkspaceLights)]],
    constant ElementPickPushConstants &pc [[buffer(BufferIndex_PushConstants)]]
) {
    const Scene scene{bindless, view, theme, workspace};

    ElementPickCandidate best;
    best.Id = 0u;
    best.Depth = 1.0f;
    best.DistanceSq = 0xffffffffu;

    const uint diameter = pc.Radius * 2u + 1u;
    const uint pixel_count = diameter * diameter;
    const uint2 extent = uint2(pc.HeadExtent);
    const int2 size = int2(extent);
    if (idx < pixel_count) {
        const int dx = int(idx % diameter) - int(pc.Radius);
        const int dy = int(idx / diameter) - int(pc.Radius);
        const uint distance_sq = uint(dx * dx + dy * dy);
        if (distance_sq <= pc.Radius * pc.Radius) {
            const int2 pixel = int2(uint2(pc.TargetPx)) + int2(dx, dy);
            if (SelectionPixelInBounds(pixel, size)) {
                uint node_idx = SelectionHeadAt(scene, pc.HeadSlot, extent, pixel);
                while (node_idx != INVALID_SELECTION_NODE) {
                    const SelectionNode node = SelectionNodeAt(scene, pc.SelectionNodesIndex, node_idx);
                    if (node.Id != 0u && (best.Id == 0u || node.Depth < best.Depth)) {
                        best.Id = node.Id;
                        best.Depth = node.Depth;
                        best.DistanceSq = distance_sq;
                    }
                    node_idx = node.Next;
                }
            }
        }
    }

    shared_candidates[local] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = ElementPickGroupSize / 2; offset > 0u; offset >>= 1u) {
        if (local < offset) {
            const ElementPickCandidate other = shared_candidates[local + offset];
            const ElementPickCandidate current = shared_candidates[local];
            if (IsBetter(other, current)) shared_candidates[local] = other;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local == 0u) {
        device ElementPickCandidate *candidates = BindlessBufferMutable(ElementPickCandidate, bindless.Buffer, pc.ElementCandidateBufferIndex);
        candidates[group_id] = shared_candidates[0];
    }
}

#endif
