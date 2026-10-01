#pragma once

#include "gpu/Types.h"

struct InsetPreviewPushConstants {
    uint32_t BasisSlot DEFAULT(InvalidSlot);
    uint32_t VertexSlot DEFAULT(InvalidSlot);
    uint32_t Count DEFAULT();
    float Thickness DEFAULT(), Depth DEFAULT();
};
static_assert(sizeof(InsetPreviewPushConstants) == 20, "InsetPreviewPushConstants size");
