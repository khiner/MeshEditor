#pragma once

#include "gpu/Types.h"

struct MorphTargetVertex {
    vec3 PositionDelta DEFAULT();
    vec3 NormalDelta DEFAULT();
};
static_assert(sizeof(MorphTargetVertex) == 24, "MorphTargetVertex size");
