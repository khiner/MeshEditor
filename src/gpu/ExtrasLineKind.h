#pragma once

#include "gpu/Types.h"

enum class ExtrasLineKind : uint32_t {
    Empty = 0,
    Camera = 1,
    LightPoint = 2,
    LightDirectional = 3,
    LightSpot = 4,
    ColliderBox = 5,
    ColliderSphere = 6,
    ColliderCylinder = 7,
    ColliderCapsule = 8,
};
