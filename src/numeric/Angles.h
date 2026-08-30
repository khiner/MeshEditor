#pragma once

#include "numeric/vec3.h"

#include <numbers>

namespace numeric {
inline constexpr float Radians(float degrees) { return degrees * (std::numbers::pi_v<float> / 180.f); }
inline constexpr vec3 Radians(vec3 degrees) { return degrees * (std::numbers::pi_v<float> / 180.f); }
inline constexpr float Degrees(float radians) { return radians * (180.f / std::numbers::pi_v<float>); }
inline constexpr vec3 Degrees(vec3 radians) { return radians * (180.f / std::numbers::pi_v<float>); }
} // namespace numeric
