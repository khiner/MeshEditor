#pragma once

#include "gpu/Types.h"

struct BoneDeformVertex {
    uvec4 Joints DEFAULT();
    vec4 Weights DEFAULT();
};
static_assert(sizeof(BoneDeformVertex) == 32, "BoneDeformVertex size");
