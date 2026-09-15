#pragma once

#include "gpu/Types.h"

// UniformFaceOffset and InvalidOffset encode uniform Face and Vertex classification without a class buffer.
enum class CornerClassEncoding : uint32_t {
    TagShift = 30,
    IndexMask = 0x3fffffffu,
    UniformFaceOffset = 0xfffffffeu,
};
