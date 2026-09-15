#pragma once

#include "gpu/Types.h"

struct DepthPyramidReducePushConstants {
    uint32_t SrcSamplerSlot DEFAULT(InvalidSlot);
    uint32_t SrcLod DEFAULT();
    uint32_t SrcWidth DEFAULT();
    uint32_t SrcHeight DEFAULT();
    GpuArray<uint32_t, 6> DstSlots DEFAULT(InvalidSlot, InvalidSlot, InvalidSlot, InvalidSlot, InvalidSlot, InvalidSlot);
};
static_assert(sizeof(DepthPyramidReducePushConstants) == 40, "DepthPyramidReducePushConstants size");
