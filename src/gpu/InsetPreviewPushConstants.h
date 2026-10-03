#pragma once

#include "gpu/SlotOffset.h"

struct InsetPreviewPushConstants {
    SlotOffset Basis DEFAULT();
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t Count DEFAULT();
    float Thickness DEFAULT(), Depth DEFAULT();
};
static_assert(sizeof(InsetPreviewPushConstants) == 24, "InsetPreviewPushConstants size");
