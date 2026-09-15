#pragma once

#include "gpu/Types.h"

// Defines deterministic compaction routes with separate fixed-function and discard-capable material visibility.
enum class MeshletRoute : uint32_t {
    OpaqueCullBack = 0,
    Blend = 1,
    Transmission = 2,
    OpaqueCullFront = 3,
    OpaqueDoubleSided = 4,
    Coverage = 5,
    EditOverlay = 6,
    Wire = 7,
    Overlay = 8,
    Count = 9,
};
