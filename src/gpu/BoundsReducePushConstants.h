#pragma once

#include "gpu/ElementWork.h"
#include "gpu/Types.h"

// Push constants shared by pose and bounds passes. Posed-buffer slots are in SceneViewUBO.
struct BoundsReducePushConstants {
    ElementWork Work DEFAULT();
    ElementWork NextWork DEFAULT();
    uint32_t EntryIndex DEFAULT();
    uint32_t BoundsEntrySlot DEFAULT();
    uint32_t BoundsSlot DEFAULT();
    uint32_t TileMapSlot DEFAULT();
    uint32_t ValuesSlot DEFAULT();
    uint32_t NodesSlot DEFAULT();
    uint32_t MembersSlot DEFAULT();
    uint32_t FirstTile DEFAULT();
    uint32_t Level DEFAULT();
};
static_assert(sizeof(BoundsReducePushConstants) == 68, "BoundsReducePushConstants size");
