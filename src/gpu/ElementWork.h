#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

struct ElementWork {
    SlotOffset Storage DEFAULT();
    uint32_t Count DEFAULT();
};
static_assert(sizeof(ElementWork) == 12, "ElementWork size");
