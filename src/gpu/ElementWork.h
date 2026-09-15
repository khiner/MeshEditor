#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"

struct ElementWork {
    SlotOffset Storage DEFAULT();
    uint32_t Count DEFAULT();
};
static_assert(sizeof(ElementWork) == 12, "ElementWork size");
