#pragma once

#include "gpu/Types.h"

// A tiled job batch's word storage slot and the word offsets of its jobs, tile map and scratch, with each pass's tiles beginning at FirstTile.
struct TiledJobPushConstants {
    uint32_t StorageSlot DEFAULT(InvalidSlot);
    uint32_t JobsOffset DEFAULT(), TileMapOffset DEFAULT(), ScratchOffset DEFAULT();
    uint32_t FirstTile DEFAULT();
};
static_assert(sizeof(TiledJobPushConstants) == 20, "TiledJobPushConstants size");
