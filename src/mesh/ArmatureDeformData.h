#pragma once

#include "numeric/vec4.h"

#include <vector>

// Optional per-vertex armature deformation channels.
struct ArmatureDeformData {
    std::vector<uvec4> Joints;
    std::vector<vec4> Weights;
};
