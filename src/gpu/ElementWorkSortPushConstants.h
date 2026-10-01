#pragma once
#include "gpu/ElementWork.h"
#include "gpu/SlotOffset.h"

struct ElementWorkSortJob {
    ElementWork Work DEFAULT();
    uint32_t TemporaryOffset DEFAULT();
};
static_assert(sizeof(ElementWorkSortJob) == 20, "ElementWorkSortJob size");

struct ElementWorkSortPushConstants {
    SlotOffset Jobs DEFAULT(); // Word offset, and temporary arrays share this buffer.
    uint32_t Shift DEFAULT();
};
static_assert(sizeof(ElementWorkSortPushConstants) == 12, "ElementWorkSortPushConstants size");
