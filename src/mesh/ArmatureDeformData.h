#pragma once

#include "numeric/vec4.h"

#include <array>
#include <cstdint>
#include <vector>

// Optional per-vertex armature deformation channels.
struct ArmatureDeformData {
    std::vector<std::array<uint32_t, 4>> Joints;
    std::vector<vec4> Weights;
};
