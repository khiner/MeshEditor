#pragma once

#include "gpu/Types.h"

// Covers a contiguous meshlet-record range with conservative bounds and maximum group error.
// Leaves reference meshlet ranges. Internal nodes reference child ranges.
struct LodNode {
    vec3 Center DEFAULT();
    float Radius DEFAULT();
    float Error DEFAULT();
    uint32_t FirstMeshlet DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t ChildOffset DEFAULT();
    uint32_t ChildCount DEFAULT();
};
static_assert(sizeof(LodNode) == 36, "LodNode size");
