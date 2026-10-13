#pragma once

#include "VectorMath.h"

#include <numbers>

namespace numeric {
constexpr float Radians(float degrees) { return degrees * (std::numbers::pi_v<float> / 180.f); }
constexpr vec3 Radians(vec3 degrees) { return degrees * (std::numbers::pi_v<float> / 180.f); }
constexpr float Degrees(float radians) { return radians * (180.f / std::numbers::pi_v<float>); }
constexpr vec3 Degrees(vec3 radians) { return radians * (180.f / std::numbers::pi_v<float>); }
} // namespace numeric
