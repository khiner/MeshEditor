#ifndef SELECTION_SHARED_MSL
#define SELECTION_SHARED_MSL

#include "Bindless.metal"
#include "SelectionCounters.metal"
#include "SelectionNode.metal"
#include "SelectionContext.metal"

// The fragment passes append nodes and swap the per-pixel head, so they reach the same buffers the
// traversal reads, through atomic views.
template<typename SetT>
inline device atomic_uint *SelectionHeadsAtomic(const thread SceneT<SetT> &scene, uint head_slot) {
    return BindlessBufferMutable(atomic_uint, scene.B.Buffer, head_slot);
}

template<typename SetT>
inline device SelectionNode *SelectionNodesMutable(const thread SceneT<SetT> &scene, uint nodes_slot) {
    return BindlessBufferMutable(SelectionNode, scene.B.Buffer, nodes_slot);
}

// SelectionCounters is (Count, Overflow), both incremented atomically.
template<typename SetT>
inline device atomic_uint *SelectionCountersAtomic(const thread SceneT<SetT> &scene, uint counter_slot) {
    return BindlessBufferMutable(atomic_uint, scene.B.Buffer, counter_slot);
}

constant uint SelectionCounterCount = 0;
constant uint SelectionCounterOverflow = 1;

// Append one fragment to this pixel's list, returning false when the node pool is full.
template<typename SetT>
inline bool SelectionAppend(const thread SceneT<SetT> &scene, SelectionContext sel, uint2 pixel, float depth, uint id) {
    device atomic_uint *counters = SelectionCountersAtomic(scene, sel.CounterSlot);
    const uint idx = atomic_fetch_add_explicit(&counters[SelectionCounterCount], 1u, memory_order_relaxed);
    if (idx >= sel.NodesCapacity) {
        atomic_fetch_add_explicit(&counters[SelectionCounterOverflow], 1u, memory_order_relaxed);
        return false;
    }
    device SelectionNode *nodes = SelectionNodesMutable(scene, sel.NodesSlot);
    nodes[idx].Depth = depth;
    nodes[idx].Id = id;

    device atomic_uint *heads = SelectionHeadsAtomic(scene, sel.HeadSlot);
    const uint head_index = pixel.y * sel.HeadExtent[0] + pixel.x;
    nodes[idx].Next = atomic_exchange_explicit(&heads[head_index], idx, memory_order_relaxed);
    return true;
}

#endif
