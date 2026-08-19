#ifndef SELECTION_TRAVERSAL_MSL
#define SELECTION_TRAVERSAL_MSL

#include "Bindless.metal"
#include "SelectionNode.metal"

constant uint INVALID_SELECTION_NODE = 0xffffffffu;

// Metal has no texture atomics, so the per-pixel linked-list heads live in a buffer indexed by
// pixel rather than a storage image.
template<typename SetT>
inline device const uint *SelectionHeads(const thread SceneT<SetT> &scene, uint head_slot) {
    return BindlessBuffer(uint, scene.B.Buffer, head_slot);
}

template<typename SetT>
inline device const SelectionNode *SelectionNodes(const thread SceneT<SetT> &scene, uint nodes_slot) {
    return BindlessBuffer(SelectionNode, scene.B.Buffer, nodes_slot);
}

inline bool SelectionPixelInBounds(int2 pixel, int2 size) {
    return pixel.x >= 0 && pixel.y >= 0 && pixel.x < size.x && pixel.y < size.y;
}

template<typename SetT>
inline uint SelectionHeadAt(const thread SceneT<SetT> &scene, uint head_slot, uint2 extent, int2 pixel) {
    return SelectionHeads(scene, head_slot)[uint(pixel.y) * extent.x + uint(pixel.x)];
}

template<typename SetT>
inline SelectionNode SelectionNodeAt(const thread SceneT<SetT> &scene, uint nodes_slot, uint node_idx) {
    return SelectionNodes(scene, nodes_slot)[node_idx];
}

inline bool SelectionIdInRange(uint id, uint max_id) { return id > 0u && id <= max_id; }

#endif
