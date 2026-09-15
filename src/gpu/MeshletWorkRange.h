#pragma once

#include "gpu/Types.h"

// Defines a producer-independent meshlet range with deterministic flattened WorkOffset ordering.
struct MeshletWorkRange {
    uint32_t Instance DEFAULT();
    uint32_t MeshletOffset DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t WorkOffset DEFAULT();
};
static_assert(sizeof(MeshletWorkRange) == 16, "MeshletWorkRange size");
