#pragma once

#include "gpu/Types.h"

// Meshlet vertices pack flat-only and canonical-owner bits above the representative corner index.
// Local triangle bytes pack all-flat above the first local vertex index.
enum class MeshletGeometryEncoding : uint32_t {
    FlatVertexBit = 0x80000000u,
    EditVertexOwnerBit = 0x40000000u,
    CornerMask = 0x3fffffffu,
    FlatTriangleBit = 0x80u,
    LocalIndexMask = 0x3fu,
    // Triangle offsets use 30 bits. Non-triangle records reuse the top two bits for MeshPrimitiveTopology.
    LocalTriangleOffsetMask = 0x3fffffffu,
    TopologyShift = 30u,
};
