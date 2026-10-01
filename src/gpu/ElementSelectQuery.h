#pragma once

#include "gpu/Types.h"

// Box queries set canonical element bits and append each word's address on its first bit.
struct SelectionQueryRef {
    uint32_t MasksSlot DEFAULT(InvalidSlot), WordsSlot DEFAULT(InvalidSlot), CountSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(SelectionQueryRef) == 12);

struct ElementSelectQuery {
    uvec4 Box DEFAULT();
    SelectionQueryRef Results DEFAULT();
    uvec2 TargetPx DEFAULT();
    uint32_t RadiusSq DEFAULT();
    uint32_t KeySlot DEFAULT(InvalidSlot);
    uint32_t IdSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(ElementSelectQuery) == 48, "ElementSelectQuery size");
