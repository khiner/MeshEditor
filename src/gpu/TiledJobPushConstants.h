#pragma once

#include "gpu/Types.h"

// The slots a tiled job batch binds, with each pass's tiles beginning at FirstTile.
struct TiledJobPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
};
static_assert(sizeof(TiledJobPushConstants) == 16, "TiledJobPushConstants size");
