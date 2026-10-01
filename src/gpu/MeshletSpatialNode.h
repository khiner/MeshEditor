#pragma once
#include "gpu/AABB.h"

// A finest meshlet owns the node at its canonical meshlet handle. Subtree
// bounds and parent links support GPU queries and local deterministic edits.
struct MeshletSpatialNode {
    AABB Box DEFAULT();
    uint32_t Parent DEFAULT(InvalidOffset);
    uint32_t Left DEFAULT(InvalidOffset), Right DEFAULT(InvalidOffset);
    uvec2 Key DEFAULT(); // Low and high words of a Morton-ordered center.
    uint32_t Meshlet DEFAULT(InvalidOffset);
    uvec2 LocalVolume DEFAULT(), SubtreeVolume DEFAULT(); // IEEE 754 doubles, packed for Metal.
};
static_assert(sizeof(MeshletSpatialNode) == 64);
