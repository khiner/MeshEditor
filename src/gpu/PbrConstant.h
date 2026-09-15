#pragma once

#include "gpu/Types.h"

// Function-constant indices for the shader that declares them.
enum class PbrConstant : uint32_t {
    EnablePunctual = 0,
    EnableTransmission = 1,
    EnableDiffuseTrans = 2,
    EnableClearcoat = 3,
    EnableSheen = 4,
    EnableAnisotropy = 5,
    EnableIridescence = 6,
    TransmissionPrepass = 7,
    NonTriangleTopology = 8,
};
