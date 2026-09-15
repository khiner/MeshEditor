#pragma once

#include "gpu/Types.h"

struct OverlayJobDrawPushConstants {
    uint32_t JobsSlot DEFAULT();
    uint32_t VisibleSlot DEFAULT();
    uint32_t InstanceSlot DEFAULT();
    uint32_t BoundsSlot DEFAULT();
    uint32_t ModelSlot DEFAULT();
    uint32_t StateSlot DEFAULT();
    uint32_t TetPositionSlot DEFAULT();
    uint32_t TetEdgeIndexSlot DEFAULT();
};
static_assert(sizeof(OverlayJobDrawPushConstants) == 32, "OverlayJobDrawPushConstants size");
