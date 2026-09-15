#pragma once

#include "gpu/Types.h"
#include "gpu/DrawData.h"
#include "gpu/SlotOffset.h"

// Stores immutable topology and shading once per mesh. Instance records contain mutable state.
struct PrimitiveRecord {
    DrawData Draw DEFAULT();
    // Optional kind-specific indices for bone adjacency and ring geometry.
    SlotOffset AuxIndices DEFAULT();
    uint32_t PrimitiveIndex DEFAULT();
    uint32_t FirstTriangle DEFAULT();
    uint32_t MeshletOffset DEFAULT();
    uint32_t MeshletCount DEFAULT();
    // Original clusters form the first Level0Count records for pinned-instance drawing.
    uint32_t Level0Count DEFAULT();
    // Identifies the full-run span root and the leaf covering the original-geometry prefix.
    uint32_t LodRootNode DEFAULT(InvalidOffset);
    uint32_t LodFinestNode DEFAULT(InvalidOffset);
};
static_assert(sizeof(PrimitiveRecord) == 264, "PrimitiveRecord size");
