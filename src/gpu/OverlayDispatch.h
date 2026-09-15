#pragma once

#include "gpu/Types.h"

enum class OverlayDispatch : uint32_t {
    LineGroupLines = 48,
    ColliderCircleSegments = 48,
    LightRangeSegments = 32,
    SpotConeSegments = 32,
    ExtrasHaloLines = 27,
    BoneSolidVertices = 24,
    BoneWireVertices = 24,
    BoneSphereVertices = 96,
    BoneSphereWireVertices = 64,
};
