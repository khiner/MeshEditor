#pragma once

#include "gpu/ElementAttributeRef.h"
#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Configures two-phase normal derivation over 256-element tiles.
// Phase 0 derives face normals. Phase 1 gathers vertex and normal-sector values through FaceNormalSlot.
// FirstTile and the buffer slots select the dispatch's tile range and posed or base storage.
struct NormalDerivePushConstants {
    ElementWork Work DEFAULT();
    uint32_t EntryIndex DEFAULT();
    uint32_t EntriesSlot DEFAULT();
    ElementAttributeRef CornerSectors DEFAULT();
    uint32_t EdgeSharpnessSlot DEFAULT();
    uint32_t FaceSharpnessSlot DEFAULT();
    uint32_t TileMapSlot DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t Phase DEFAULT();
    uint32_t PositionSlot DEFAULT();
    uint32_t PositionNodesSlot DEFAULT();
    uint32_t VertexNormalSlot DEFAULT();
    uint32_t VertexNormalNodesSlot DEFAULT();
    ElementAttributeRef NormalSectors DEFAULT();
    uint32_t PosedSectorNodesSlot DEFAULT();
    uint32_t PosedSectorValuesSlot DEFAULT();
    uint32_t FaceNormalSlot DEFAULT();
    uint32_t FaceNormalNodesSlot DEFAULT();
    uint32_t BaseFaceNormalSlot DEFAULT();
};
static_assert(sizeof(NormalDerivePushConstants) == 96, "NormalDerivePushConstants size");
