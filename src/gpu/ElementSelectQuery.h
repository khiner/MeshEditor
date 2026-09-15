#pragma once

#include "gpu/Types.h"

struct ElementSelectQuery {
    uvec4 Box DEFAULT();
    uint32_t BoxResultSlot DEFAULT(InvalidSlot);
    uvec2 TargetPx DEFAULT();
    uint32_t RadiusSq DEFAULT();
    uint32_t KeySlot DEFAULT(InvalidSlot);
    uint32_t IdSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(ElementSelectQuery) == 40, "ElementSelectQuery size");
