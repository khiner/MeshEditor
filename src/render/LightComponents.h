#pragma once

#include "Field.h"
#include "gpu/LightType.h"
#include "render/RenderSchema.h"

// Punctual intensities use glTF units; area lights use luminous flux in lumens.
// Range 0 is infinite, and cone angles are radians.
struct SceneLight {
    float Range{0.f};
    vec3 Color{1.f, 1.f, 1.f};
    float Intensity{1.f};
    float InnerConeAngle{0.f}, OuterConeAngle{0.f};
    LightType Type{LightType::Point};
    vec2 Size{1.f, 1.f};

    bool operator==(const SceneLight &) const = default;
};
template<> inline constexpr field::SpecTable<SceneLight> Specs<SceneLight>{
    {&SceneLight::Range, {.Min = 0, .Max = 1000, .Unit = FieldUnit::Meters, .Help = "Distance at which the light fades to zero. Zero is infinite."}},
    {&SceneLight::Color, {.Widget = FieldWidget::Color, .Help = "Color of the emitted light"}},
    {&SceneLight::Intensity, {.Min = 0, .Max = 1000, .Help = "Brightness in candela for point and spot lights, lux for directional lights, and lumens for area lights"}},
    {&SceneLight::InnerConeAngle, {.Unit = FieldUnit::Radians, .Help = "Angle from the spot axis where the falloff begins"}},
    {&SceneLight::OuterConeAngle, {.Unit = FieldUnit::Radians, .Help = "Angle from the spot axis where the light ends"}},
    {&SceneLight::Size, {.Min = 0, .Max = 1000, .Speed = 0.01f, .Unit = FieldUnit::Meters, .Help = "Full local width and height of the emitter. Object scale also affects its size; total emitted flux stays constant."}},
    {&SceneLight::Type, {.Help = "Directional lights shine from infinitely far away, point lights in all directions, and spot lights in a cone; rectangles and ellipses emit from a finite surface"}},
};

// The scene's EXT_lights_image_based light on the document entity.
// Rotation turns the environment and Intensity scales it.
struct ImageLight {
    quat Rotation{1, 0, 0, 0};
    float Intensity{1.f};

    bool operator==(const ImageLight &) const = default;
};
template<> inline constexpr field::SpecTable<ImageLight> Specs<ImageLight>{{&ImageLight::Intensity, {.Min = 0, .Max = 2, .Help = "Multiplier on the scene's image-based lighting"}}};
