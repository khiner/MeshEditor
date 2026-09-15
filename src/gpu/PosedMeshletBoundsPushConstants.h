#pragma once

#include "gpu/Types.h"
#include "gpu/ElementWork.h"

// Configures one posed-entry meshlet AABB reduction.
struct PosedMeshletBoundsPushConstants {
    ElementWork Work DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t DrawDataSlot DEFAULT();
    uint32_t TileMapSlot DEFAULT();
    uint32_t MeshletSlot DEFAULT();
    uint32_t PrimitiveSlot DEFAULT();
    uint32_t MeshletVertexSlot DEFAULT();
    uint32_t PosedMeshletBoundsSlot DEFAULT();
};
static_assert(sizeof(PosedMeshletBoundsPushConstants) == 40, "PosedMeshletBoundsPushConstants size");
