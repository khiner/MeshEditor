#pragma once

#include "gpu/Types.h"

// One triangle of a cluster, as three local vertex bytes carrying the MeshletGeometryEncoding flags.
using MeshletTriangle = GpuArray<uint8_t, 3>;
static_assert(sizeof(MeshletTriangle) == 3, "MeshletTriangle size");

// The bases of a record's ranges in the cluster payload pools, which its clusters' payload offsets are relative to.
struct MeshletPayloadBases {
    uint32_t TriangleIds DEFAULT(), VertexCorners DEFAULT(), LocalTriangles DEFAULT();
};
static_assert(sizeof(MeshletPayloadBases) == 12, "MeshletPayloadBases size");

// The slots of the cluster payload pools.
struct MeshletPayloadSlots {
    uint32_t TriangleIds DEFAULT(InvalidSlot), VertexCorners DEFAULT(InvalidSlot), LocalTriangles DEFAULT(InvalidSlot);
};
static_assert(sizeof(MeshletPayloadSlots) == 12, "MeshletPayloadSlots size");
