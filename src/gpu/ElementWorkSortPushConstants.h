#pragma once
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

struct ElementWorkSortJob {
    ElementWork Work DEFAULT();
    uint32_t TemporaryOffset DEFAULT();
};

struct ElementWorkSortPushConstants {
    SlotOffset Jobs DEFAULT(); // Word offset, and temporary arrays share this buffer.
    uint32_t Shift DEFAULT();
};
