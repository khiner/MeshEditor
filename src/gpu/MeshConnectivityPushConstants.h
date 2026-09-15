#pragma once

#include "gpu/Types.h"

// Shared connectivity slots with per-pass tiles beginning at FirstTile.
struct MeshConnectivityPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
};
static_assert(sizeof(MeshConnectivityPushConstants) == 16, "MeshConnectivityPushConstants size");
