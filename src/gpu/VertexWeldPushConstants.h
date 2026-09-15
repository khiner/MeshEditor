#pragma once

#include "gpu/Types.h"

// Shared vertex-weld slots with per-pass tiles beginning at FirstTile.
struct VertexWeldPushConstants {
    uint32_t JobsSlot DEFAULT(InvalidSlot);
    uint32_t TileMapSlot DEFAULT(InvalidSlot);
    uint32_t ScratchSlot DEFAULT(InvalidSlot);
    uint32_t FirstTile DEFAULT();
};
static_assert(sizeof(VertexWeldPushConstants) == 16, "VertexWeldPushConstants size");
