#pragma once

#include "gpu/Types.h"

// Conservative bounds and maximum group error over a subtree.
// Leaves own sparse cluster membership.
// Internal nodes reference child ranges.
// FirstMeshlet is construction metadata, not the live draw address mapping.
struct LodNode {
    vec3 Center DEFAULT();
    float Radius DEFAULT();
    float Error DEFAULT();
    uint32_t FirstMeshlet DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t ChildOffset DEFAULT();
    uint32_t ChildCount DEFAULT();
    uint32_t MeshletRoot DEFAULT(InvalidOffset);
};
static_assert(sizeof(LodNode) == 40, "LodNode size");
