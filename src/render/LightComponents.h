#pragma once

#include "gpu/PunctualLightType.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <cstdint>

// A KHR_lights_punctual light in glTF units. Range 0 is infinite, and cone angles are radians.
struct PunctualLight {
    float Range{0.f};
    vec3 Color{1.f, 1.f, 1.f};
    float Intensity{1.f};
    float InnerConeAngle{0.f}, OuterConeAngle{0.f};
    PunctualLightType Type{PunctualLightType::Point};

    bool operator==(const PunctualLight &) const = default;
};

// The scene's EXT_lights_image_based light on the viewport. Rotation turns the environment and Intensity scales it.
struct ImageLight {
    quat Rotation{1, 0, 0, 0};
    float Intensity{1.f};

    bool operator==(const ImageLight &) const = default;
};

// The light's slot in the GPU light buffer.
struct LightIndex {
    uint32_t Value{0};
};
