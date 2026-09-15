#pragma once

#include "gpu/Types.h"
#include "gpu/SlotOffset.h"
#include "gpu/PunctualLightType.h"

struct PunctualLight {
    SlotOffset TransformSlotOffset DEFAULT();
    float Range DEFAULT();
    vec3 Color DEFAULT();
    float Intensity DEFAULT();
    float InnerConeCos DEFAULT();
    float OuterConeCos DEFAULT();
    PunctualLightType Type DEFAULT();
};
static_assert(sizeof(PunctualLight) == 40, "PunctualLight size");
