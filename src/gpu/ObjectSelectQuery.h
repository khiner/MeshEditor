#pragma once

#include "gpu/Types.h"

struct ObjectSelectQuery {
    uint32_t MaxId DEFAULT();
    uvec2 TargetPx DEFAULT();
    uint32_t RadiusSq DEFAULT();
    uint32_t EpochInv DEFAULT();
    uint32_t BestKeySlot DEFAULT(InvalidSlot);
    uint32_t SeenBitsSlot DEFAULT(InvalidSlot);
    uvec4 Box DEFAULT();
    uint32_t BoxResultSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(ObjectSelectQuery) == 48, "ObjectSelectQuery size");
