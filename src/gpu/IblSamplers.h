#pragma once

#include "gpu/Types.h"

struct IblSamplers {
    uint32_t DiffuseEnvSamplerSlot DEFAULT();
    uint32_t SpecularEnvSamplerSlot DEFAULT();
    uint32_t SpecularEnvMipCount DEFAULT();
};
