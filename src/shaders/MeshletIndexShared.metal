#ifndef MESHLET_INDEX_SHARED_MSL
#define MESHLET_INDEX_SHARED_MSL
#include "Bindless.metal"
#include "gpu/MeshletIndex.h"

inline uint MeshletIndexSelect(device const BindlessSet &b, MeshletIndexRef ref, uint rank) {
    if (ref.Root == InvalidOffset) return InvalidOffset;
    device const MeshletIndexNode *nodes = BindlessBuffer(MeshletIndexNode,b.Buffer,ref.NodesSlot);
    device const MeshletIndexLeaf *leaves = BindlessBuffer(MeshletIndexLeaf,b.Buffer,ref.LeavesSlot);
    uint id = ref.Root;
    if (rank >= nodes[id].Count) return InvalidOffset;
    if (nodes[id].DenseFirst != InvalidOffset) return nodes[id].DenseFirst + rank;
    uint depth = nodes[id].SelectDepth;
    id = nodes[id].SelectNode;
    while (depth) {
        if (nodes[id].DenseFirst != InvalidOffset) return nodes[id].DenseFirst + rank;
        const uint active = nodes[id].Active;
        const uint first = ctz(active);
        uint lo = first, hi = 32u-clz(active);
        while (lo+1u < hi) {
            const uint mid = (lo+hi)/2u;
            if (rank < nodes[id].Ends[mid-1u]) hi = mid;
            else lo = mid;
        }
        if (lo != first) rank -= nodes[id].Ends[lo-1u];
        const uint child = nodes[id].Children[lo];
        if (depth == 1u) { id = child; break; }
        depth = nodes[child].SelectDepth;
        id = nodes[child].SelectNode;
    }
    if (leaves[id].DenseFirst != InvalidOffset) return leaves[id].DenseFirst + rank;
    return SelectLiveElement(&leaves[id].Live[0], leaves[id].Block, rank);
}

inline uint MeshletIndexRank(device const BindlessSet &b, MeshletIndexRef ref, uint handle) {
    if (ref.Root == InvalidOffset || handle == InvalidOffset) return InvalidOffset;
    device const MeshletIndexNode *nodes = BindlessBuffer(MeshletIndexNode,b.Buffer,ref.NodesSlot);
    device const MeshletIndexLeaf *leaves = BindlessBuffer(MeshletIndexLeaf,b.Buffer,ref.LeavesSlot);
    uint id = ref.Root, rank = 0u, block = handle / 256u;
    for (uint level = MeshletIndexLevels; level--;) {
        const uint slot = (block >> (level * 5u)) & 31u;
        if (!(nodes[id].Active & (1u << slot))) return InvalidOffset;
        if (slot) rank += nodes[id].Ends[slot-1u];
        id = nodes[id].Children[slot];
    }
    const uint word = (handle % 256u) / 32u, bit = handle % 32u;
    if (!(leaves[id].Live[word] & (1u << bit))) return InvalidOffset;
    for (uint w = 0u; w < word; ++w) rank += popcount(leaves[id].Live[w]);
    return rank + popcount(leaves[id].Live[word] & ((1u << bit) - 1u));
}
#endif
