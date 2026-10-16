#pragma once

#include "gpu/Types.h"

// Stores one traversal block's child nodes and leaf meshlet records.
struct LodFrontierBlockState {
    uint32_t NodeCount DEFAULT();
    uint32_t MeshletCount DEFAULT();
};
