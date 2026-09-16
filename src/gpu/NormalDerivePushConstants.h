#pragma once

#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Configures two-phase normal derivation over 256-element tiles.
// Phase 0 derives face normals. Phase 1 gathers vertex and normal-sector values through FaceNormalSlot.
// FirstTile and the buffer slots select the dispatch's tile range and posed or base storage.
struct NormalDerivePushConstants {
    ElementWork Work DEFAULT();
    uint32_t EntryIndex DEFAULT();
    uint32_t EntriesSlot DEFAULT();
    uint32_t AdjacencySlot DEFAULT();
    uint32_t TileMapSlot DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t Phase DEFAULT();
    uint32_t FaceFirstTriangleSlot DEFAULT();
    uint32_t PositionSlot DEFAULT();
    uint32_t VertexNormalSlot DEFAULT();
    uint32_t SeamNormalSlot DEFAULT();
    uint32_t FaceNormalSlot DEFAULT();
};
static_assert(sizeof(NormalDerivePushConstants) == 56, "NormalDerivePushConstants size");
