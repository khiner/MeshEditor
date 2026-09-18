#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Stores one primitive's meshlet range and LOD roots. The instance record names the mesh record they share.
struct PrimitiveRecord {
    // Optional kind-specific indices for bone adjacency and ring geometry.
    SlotOffset AuxIndices DEFAULT();
    uint32_t PrimitiveIndex DEFAULT();
    // The mesh's primitive-material offset, kept here so the cull resolves a material without the mesh record.
    uint32_t PrimitiveMaterialOffset DEFAULT(InvalidOffset);
    uint32_t FirstTriangle DEFAULT();
    uint32_t MeshletOffset DEFAULT();
    uint32_t MeshletCount DEFAULT();
    // Original clusters form the first Level0Count records for pinned-instance drawing.
    uint32_t Level0Count DEFAULT();
    // Identifies the full-run span root and the leaf covering the original-geometry prefix.
    uint32_t LodRootNode DEFAULT(InvalidOffset);
    uint32_t LodFinestNode DEFAULT(InvalidOffset);
};
static_assert(sizeof(PrimitiveRecord) == 40, "PrimitiveRecord size");
