#pragma once

#include "gpu/Types.h"

// Packs canonical edit edge and reversed-winding flag. All bits set denotes an internal diagonal.
// Stores only the first meshlet occurrence of each canonical edge.
enum class MeshletEditEdgeEncoding : uint32_t {
    ReversedBit = 0x80000000u,
    EdgeMask = 0x7fffffffu,
};
