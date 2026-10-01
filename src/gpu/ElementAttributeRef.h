#pragma once
#include "gpu/Types.h"

// Optional payload blocks addressed by canonical element handles.
// A zero table entry is absent.
// Other entries hold the payload block index plus one.
struct ElementAttributeRef {
    uint32_t BlocksSlot DEFAULT(InvalidSlot);
    uint32_t ValuesSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(ElementAttributeRef) == 8);
