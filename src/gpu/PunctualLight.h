#pragma once

#include "gpu/PunctualLightType.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

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
