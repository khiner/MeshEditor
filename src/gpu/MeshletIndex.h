#pragma once
#include "gpu/Types.h"

GPU_CONSTANT uint32_t MeshletIndexLevels = 5u;

// Each root owns its leaves, so independent primitive/LOD sets may occupy the
// same canonical 256-handle block. No compact array of live handles is stored.
struct MeshletIndexNode {
    // Ownership walks read this mask and its child pointers together.
    uint32_t Active DEFAULT();
    GpuArray<uint32_t, 32> Children DEFAULT();
    // Inclusive child populations.
    // Local updates refresh one SIMD-sized node.
    // Readers select a child without scanning earlier siblings.
    GpuArray<uint32_t, 32> Ends DEFAULT();
    uint32_t Parent DEFAULT(InvalidOffset), Count DEFAULT();
    // Skip chains with one live child when selecting by rank.
    // Depth zero names a leaf.
    // Otherwise SelectNode names the next branching node.
    uint32_t SelectNode DEFAULT(InvalidOffset), SelectDepth DEFAULT();
    // First canonical handle when every live handle below this node is a consecutive run.
    // Local insertions/removals refresh it bottom-up.
    uint32_t DenseFirst DEFAULT(InvalidOffset);
};
static_assert(sizeof(MeshletIndexNode) == 280);

struct MeshletIndexLeaf {
    GpuArray<uint32_t, 8> Live DEFAULT();
    uint32_t Parent DEFAULT(InvalidOffset), Block DEFAULT(), Count DEFAULT(), DenseFirst DEFAULT(InvalidOffset);
};
static_assert(sizeof(MeshletIndexLeaf) == 48);

struct MeshletIndexRef {
    uint32_t NodesSlot DEFAULT(InvalidSlot), LeavesSlot DEFAULT(InvalidSlot), Root DEFAULT(InvalidOffset);
};
static_assert(sizeof(MeshletIndexRef) == 12);
