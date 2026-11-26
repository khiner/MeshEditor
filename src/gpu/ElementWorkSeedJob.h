#pragma once
#include "gpu/ElementWork.h"

// Each listed block contributes its live elements with handles from First below End, masked by a selection when one is named.
struct ElementWorkSeedJob {
    ElementWork Work DEFAULT();
    SlotOffset BlockIds DEFAULT();
    uint32_t BlockCount DEFAULT();
    uint32_t BlocksSlot DEFAULT(), Owner DEFAULT();
    uint32_t SelectionSlot DEFAULT(InvalidSlot);
    uint32_t First DEFAULT(), End DEFAULT(InvalidOffset);
};
