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
    // Construction slice in the canonical triangle-ID render arena.
    uint32_t TriangleOffset DEFAULT(), TriangleCount DEFAULT();
    uint32_t MeshletCount DEFAULT();
    // Pinned instances enumerate the finest node's canonical membership.
    uint32_t Level0Count DEFAULT();
    // Identifies the full-run span root and the leaf covering the original-geometry prefix.
    uint32_t LodRootNode DEFAULT(InvalidOffset);
    uint32_t LodFinestNode DEFAULT(InvalidOffset);
    // The whole-primitive simplification scale that every rebuild of one of its LOD groups reuses.
    float SimplifyScale DEFAULT();
    // Channels the current hierarchy preserves for material and debug shading.
    uint32_t LodAttributes DEFAULT(~0u);
};
static_assert(sizeof(PrimitiveRecord) == 48, "PrimitiveRecord size");
