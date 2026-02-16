#pragma once

#include "numeric/vec3.h"

#include <optional>
#include <utility>
#include <variant>

struct DirectionalLight {};

struct PointLight {
    std::optional<float> Range; // nullopt = infinite, 0 = disabled (no contribution)
};

struct SpotLight {
    std::optional<float> Range; // nullopt = infinite, 0 = disabled (no contribution)
    float Size; // radians (matches glTF outerConeAngle)
    float Blend; // [0, 1], where inner = size * (1 - blend)
};
using LightVariant = std::variant<DirectionalLight, PointLight, SpotLight>;

struct LightData {
    LightVariant Type;
    vec3 Color; // Linear RGB
    float Intensity; // Candela (point/spot) or lux (directional)
};
