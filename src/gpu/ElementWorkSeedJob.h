#pragma once
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

// Each listed block contributes its live elements, masked by a selection when one is named.
struct ElementWorkSeedJob {
    ElementWork Work DEFAULT();
    SlotOffset BlockIds DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t BlocksSlot DEFAULT(), Owner DEFAULT();
    uint32_t SelectionSlot DEFAULT(InvalidSlot);
};
static_assert(sizeof(ElementWorkSeedJob) == 40);
