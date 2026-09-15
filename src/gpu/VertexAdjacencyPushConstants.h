#pragma once

#include "gpu/Types.h"

// Shared vertex-adjacency slots with per-pass tiles beginning at FirstTile.
struct VertexAdjacencyPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t AdjacencySlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
};
static_assert(sizeof(VertexAdjacencyPushConstants) == 20, "VertexAdjacencyPushConstants size");
