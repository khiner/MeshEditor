#pragma once

#include "gpu/Types.h"

enum class LightType : uint32_t {
    Directional = 0,
    Point = 1,
    Spot = 2,
    Rectangle = 3,
    Ellipse = 4,
};

constexpr bool IsAreaLight(LightType type) { return type == LightType::Rectangle || type == LightType::Ellipse; }
