#pragma once

#include "gpu/PunctualLightType.h"
#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// The GPU light record, uploaded from the PunctualLight component with its spot cone angles as cosines.
struct LightRecord {
    SlotOffset TransformSlotOffset DEFAULT();
    float Range DEFAULT();
    vec3 Color DEFAULT();
    float Intensity DEFAULT();
    float InnerConeCos DEFAULT();
    float OuterConeCos DEFAULT();
    PunctualLightType Type DEFAULT();
};
static_assert(sizeof(LightRecord) == 40, "LightRecord size");
