#pragma once

#include "gpu/Types.h"

struct IblSamplers {
    uint32_t DiffuseEnvSamplerSlot DEFAULT();
    uint32_t SpecularEnvSamplerSlot DEFAULT();
    uint32_t BrdfLutSamplerSlot DEFAULT();
    uint32_t SpecularEnvMipCount DEFAULT();
    uint32_t SheenEnvSamplerSlot DEFAULT();
    uint32_t SheenEnvMipCount DEFAULT();
    uint32_t SheenELutSamplerSlot DEFAULT();
    uint32_t CharlieLutSamplerSlot DEFAULT();
};
static_assert(sizeof(IblSamplers) == 32, "IblSamplers size");
