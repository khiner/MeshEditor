#pragma once

#include "Field.h"
#include "gpu/PunctualLightType.h"
#include "render/RenderSchema.h"

// A KHR_lights_punctual light in glTF units.
// Range 0 is infinite, and cone angles are radians.
struct PunctualLight {
    float Range{0.f};
    vec3 Color{1.f, 1.f, 1.f};
    float Intensity{1.f};
    float InnerConeAngle{0.f}, OuterConeAngle{0.f};
    PunctualLightType Type{PunctualLightType::Point};

    bool operator==(const PunctualLight &) const = default;
};
template<> inline constexpr field::SpecTable<PunctualLight> Specs<PunctualLight>{
    {&PunctualLight::Range, {.Min = 0, .Max = 1000, .Unit = FieldUnit::Meters, .Help = "Distance at which the light fades to zero. Zero is infinite."}},
    {&PunctualLight::Color, {.Widget = FieldWidget::Color, .Help = "Color of the emitted light"}},
    {&PunctualLight::Intensity, {.Min = 0, .Max = 1000, .Help = "Brightness in candela for point and spot lights, and lux for directional lights"}},
    {&PunctualLight::InnerConeAngle, {.Unit = FieldUnit::Radians, .Help = "Angle from the spot axis where the falloff begins"}},
    {&PunctualLight::OuterConeAngle, {.Unit = FieldUnit::Radians, .Help = "Angle from the spot axis where the light ends"}},
    {&PunctualLight::Type, {.Help = "Directional lights shine from infinitely far away, point lights in all directions, and spot lights in a cone"}},
};

// The scene's EXT_lights_image_based light on the document entity.
// Rotation turns the environment and Intensity scales it.
struct ImageLight {
    quat Rotation{1, 0, 0, 0};
    float Intensity{1.f};

    bool operator==(const ImageLight &) const = default;
};
template<> inline constexpr field::SpecTable<ImageLight> Specs<ImageLight>{{&ImageLight::Intensity, {.Min = 0, .Max = 2, .Help = "Multiplier on the scene's image-based lighting"}}};
