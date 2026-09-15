#pragma once

#include "gpu/Types.h"

// Packs loop position above face index for Blender-style corner-angle-weighted normal accumulation.
// Supports up to 4.2 million faces per mesh and face valence up to 1024.
enum class FanItemEncoding : uint32_t {
    LoopShift = 22,
    FaceMask = 0x3fffffu,
};
