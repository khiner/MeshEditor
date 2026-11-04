#pragma once
// Manifold's collider: Morton order, Karras' radix tree over the sorted leaves, refit boxes and a box traversal.
#include "Bindless.metal"
#include "RadixSort.metal"
#include "gpu/Collider.h"

// A record's radix tree over its triangle boxes: leaves at even nodes, internal nodes at odd ones, and the root at node 1.
struct BoolTree {
    device const float *Boxes; // Six floats per node: minimum then maximum
    device const packed_uint2 *Children;
    device const uint *Order; // Leaf -> the record's triangle
    uint Leaves;
    uint First; // The record's first combined triangle
};
template<typename Overlaps, typename Visit>
inline void BoolTraverse(BoolTree tree, Overlaps overlaps, Visit visit) {
    if (tree.Leaves < 2u) {
        if (tree.Leaves == 1u && overlaps(tree.Boxes)) visit(tree.First + tree.Order[0]);
        return;
    }
    uint stack[64];
    int top = -1;
    uint node = 1u;
    while (true) {
        const uint2 children = uint2(tree.Children[(node - 1u) / 2u]);
        bool descend[2];
        for (uint c = 0u; c < 2u; ++c) {
            const uint child = children[c];
            const bool hit = overlaps(tree.Boxes + 6u * child);
            if (hit && (child & 1u) == 0u) visit(tree.First + tree.Order[child / 2u]);
            descend[c] = hit && (child & 1u) != 0u;
        }
        if (!descend[0] && !descend[1]) {
            if (top < 0) break;
            node = stack[top--];
        } else {
            node = descend[0] ? children[0] : children[1];
            if (descend[0] && descend[1]) stack[++top] = children[1];
        }
    }
}
inline bool BoxOverlaps(device const float *box, float3 low, float3 high) {
    return box[0] <= high.x && box[1] <= high.y && box[2] <= high.z && low.x <= box[3] && low.y <= box[4] && low.z <= box[5];
}
inline bool BoxContainsXY(device const float *box, float2 p) { return p.x >= box[0] && p.x <= box[3] && p.y >= box[1] && p.y <= box[4]; }

inline uint SpreadBits3(uint v) {
    v = 0xFF0000FFu & (v * 0x00010001u);
    v = 0x0F00F00Fu & (v * 0x00000101u);
    v = 0xC30C30C3u & (v * 0x00000011u);
    v = 0x49249249u & (v * 0x00000005u);
    return v;
}

struct BoolRadixTree {
    device const uint *Keys, *Order;
    int Leaves;
    int Prefix(int i, int j) const {
        if (j < 0 || j >= Leaves) return -1;
        const uint a = Keys[Order[i]], c = Keys[Order[j]];
        return a == c ? 32 + int(clz(uint(i) ^ uint(j))) : int(clz(a ^ c));
    }
    int RangeEnd(int i) const {
        int dir = Prefix(i, i + 1) - Prefix(i, i - 1);
        dir = (dir > 0) - (dir < 0);
        const int common = Prefix(i, i - dir);
        int max_length = 128;
        while (Prefix(i, i + dir * max_length) > common) max_length *= 4;
        int length = 0;
        for (int step = max_length / 2; step > 0; step /= 2)
            if (Prefix(i, i + dir * (length + step)) > common) length += step;
        return i + dir * length;
    }
    int FindSplit(int first, int last) const {
        const int common = Prefix(first, last);
        int split = first, step = last - first;
        do {
            step = (step + 1) >> 1;
            const int next = split + step;
            if (next < last && Prefix(first, next) > common) split = next;
        } while (step > 1);
        return split;
    }
};

inline void RadixTreeNode(BoolRadixTree tree, uint internal, device packed_uint2 *children, device uint *parents, device uint *counters) {
    const int end = tree.RangeEnd(int(internal)), first = min(int(internal), end), last = max(int(internal), end);
    int split = tree.FindSplit(first, last);
    const uint child0 = split == first ? 2u * uint(split) : 2u * uint(split) + 1u;
    ++split;
    const uint child1 = split == last ? 2u * uint(split) : 2u * uint(split) + 1u;
    children[internal] = packed_uint2(child0, child1);
    parents[child0] = parents[child1] = 2u * internal + 1u;
    counters[internal] = 0u;
}
// Writes a leaf's box and climbs, where the second child to arrive at a node writes its union.
inline void RefitClimb(JobRuns storage, ColliderRuns runs, uint leaf, float3 low, float3 high) {
    device atomic_uint *boxes = storage.Run<atomic_uint>(runs.NodeBoxes), *counters = storage.Run<atomic_uint>(runs.Counters);
    device const uint *parents = storage.Run<uint>(runs.NodeParents);
    device const packed_uint2 *children = storage.Run<packed_uint2>(runs.Children);
    const auto store = [&](uint node) {
        for (uint k = 0u; k < 3u; ++k) {
            atomic_store_explicit(&boxes[6u * node + k], as_type<uint>(low[k]), memory_order_relaxed);
            atomic_store_explicit(&boxes[6u * node + 3u + k], as_type<uint>(high[k]), memory_order_relaxed);
        }
    };
    uint node = 2u * leaf;
    store(node);
    if (runs.Leaves < 2u) return;
    do {
        node = parents[node];
        const uint internal = (node - 1u) / 2u;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        if (atomic_fetch_add_explicit(&counters[internal], 1u, memory_order_relaxed) == 0u) return;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint2 pair = uint2(children[internal]);
        for (uint k = 0u; k < 3u; ++k) {
            low[k] = min(as_type<float>(atomic_load_explicit(&boxes[6u * pair.x + k], memory_order_relaxed)), as_type<float>(atomic_load_explicit(&boxes[6u * pair.y + k], memory_order_relaxed)));
            high[k] = max(as_type<float>(atomic_load_explicit(&boxes[6u * pair.x + 3u + k], memory_order_relaxed)), as_type<float>(atomic_load_explicit(&boxes[6u * pair.y + 3u + k], memory_order_relaxed)));
        }
        store(node);
    } while (node != 1u);
}

inline BoolTree TreeOf(JobRuns storage, ColliderRuns runs, uint first) {
    return {storage.Run<float>(runs.NodeBoxes), storage.Run<packed_uint2>(runs.Children), storage.Run<uint>(runs.Order), runs.Leaves, first};
}
inline RadixSortView ColliderSort(JobRuns storage, ColliderRuns runs, uint shift) {
    return {storage.Run<uint>(runs.Keys), storage.Run<uint>(runs.Order), storage.Run<uint>(runs.Temporary), storage.Run<uint>(runs.Histogram), storage.Run<uint>(runs.Totals),
            runs.Leaves, (runs.Leaves + 255u) / 256u, 1u, 0u, shift, (shift & 4u) != 0u};
}
struct ColliderContext {
    JobRuns Storage;
    ColliderRuns Runs;
    ColliderContext(device const BindlessSet &b, constant JobPushConstants &pc, uint layer)
        : Storage{BindlessBufferMutable(uint, b.Buffer, pc.Job.Slot)}, Runs(Storage.Run<ColliderRuns>(pc.Job.Offset)[layer]) {}
};

kernel void ColliderSortHistogram(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const ColliderContext ctx{b, pc, group.z};
    const RadixSortView sort = ColliderSort(ctx.Storage, ctx.Runs, pc.Parameter);
    if (group.x >= sort.Blocks) return;
    threadgroup atomic_uint counts[16];
    RadixHistogram(sort, lane, group.x, counts);
}
kernel void ColliderSortPrefix(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    const ColliderContext ctx{b, pc, group.z};
    threadgroup uint sums[9];
    RadixPrefix(ColliderSort(ctx.Storage, ctx.Runs, pc.Parameter), lane, group.x, sl, sg, sums);
}
kernel void ColliderSortScatter(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]
) {
    const ColliderContext ctx{b, pc, group.z};
    const RadixSortView sort = ColliderSort(ctx.Storage, ctx.Runs, pc.Parameter);
    if (group.x >= sort.Blocks) return;
    threadgroup uint groups[128];
    RadixScatter(sort, lane, group.x, sl, sg, groups);
}
// One thread per internal node.
kernel void ColliderTree(
    device const BindlessSet &b [[buffer(BufferIndex_Bindless)]],
    constant JobPushConstants &pc [[buffer(BufferIndex_PushConstants)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]
) {
    const ColliderContext ctx{b, pc, group.z};
    const ColliderRuns runs = ctx.Runs;
    const uint internal = group.x * 256u + lane;
    if (runs.Leaves < 2u || internal + 1u >= runs.Leaves) return;
    const JobRuns storage = ctx.Storage;
    const BoolRadixTree tree{storage.Run<uint>(runs.Keys), storage.Run<uint>(runs.Order), int(runs.Leaves)};
    RadixTreeNode(tree, internal, storage.Run<packed_uint2>(runs.Children), storage.Run<uint>(runs.NodeParents), storage.Run<uint>(runs.Counters));
}
