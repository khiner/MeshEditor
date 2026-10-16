#pragma once

#include "gpu/Types.h"

// Conservative bounds and maximum group error over a subtree.
// A leaf lists its MeshletCount clusters as the run of member handles from Members, and InvalidOffset lists none.
// An internal node names its children as the run of node handles from ChildOffset.
// FirstMeshlet is construction metadata, not the live draw address mapping.
struct LodNode {
    vec3 Center DEFAULT();
    float Radius DEFAULT();
    float Error DEFAULT();
    uint32_t FirstMeshlet DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t ChildOffset DEFAULT();
    uint32_t ChildCount DEFAULT();
    uint32_t Members DEFAULT(InvalidOffset);
};
