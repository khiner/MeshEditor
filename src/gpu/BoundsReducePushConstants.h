#pragma once

#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Push constants shared by pose and bounds passes. Posed-buffer slots are in SceneViewUBO.
struct BoundsReducePushConstants {
    ElementWork Work DEFAULT();
    ElementWork NextWork DEFAULT();
    uint32_t EntryIndex DEFAULT();
    uint32_t DrawDataSlot DEFAULT();
    uint32_t BoundsSlot DEFAULT();
    uint32_t TileMapSlot DEFAULT();
    uint32_t PartialBoundsSlot DEFAULT();
    uint32_t EntryFirstTileSlot DEFAULT();
};
static_assert(sizeof(BoundsReducePushConstants) == 48, "BoundsReducePushConstants size");
