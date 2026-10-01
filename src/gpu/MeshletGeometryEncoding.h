#pragma once

#include "gpu/Types.h"

// Local triangle bytes pack all-flat on the first corner and physical boundary
// provenance on each directed edge. Rendering masks both from local indices.
enum class MeshletGeometryEncoding : uint32_t {
    FlatTriangleBit = 0x80u,
    PhysicalBoundaryBit = 0x40u,
    LocalIndexMask = 0x3fu,
};
