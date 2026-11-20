#pragma once

#include "gpu/LightType.h"
#include "gpu/SlotOffset.h"

// The GPU light record, uploaded from the SceneLight component with its spot cone angles as cosines.
struct LightRecord {
    SlotOffset TransformSlotOffset DEFAULT();
    float Range DEFAULT();
    vec3 Color DEFAULT();
    float Intensity DEFAULT();
    float InnerConeCos DEFAULT();
    float OuterConeCos DEFAULT();
    LightType Type DEFAULT();
    vec2 Size DEFAULT();
};

// Derived once per light and shutter pose, following EEVEE Light::sync preparation.
struct PreparedAreaLight {
    vec3 X DEFAULT(), Y DEFAULT(), Z DEFAULT();
    float Radiance DEFAULT();
    uint32_t Valid DEFAULT();
    vec2 NoiseRotation DEFAULT();
};
