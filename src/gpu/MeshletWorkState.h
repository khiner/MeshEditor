#pragma once

#include "gpu/Types.h"

struct MeshletWorkState {
    uint32_t RangeCount DEFAULT();
    uint32_t MeshletCount DEFAULT();
    uint32_t CullBlockCount DEFAULT();
};
static_assert(sizeof(MeshletWorkState) == 12, "MeshletWorkState size");
