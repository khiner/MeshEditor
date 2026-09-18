#pragma once

#include "numeric/VectorOps.h"
#include "numeric/uvec2.h"
#include "numeric/uvec4.h"
#include "numeric/vec2.h"
#include "numeric/vec4.h"

namespace numeric {
template<detail::FloatingVector V> constexpr V Mix(V a, V b, typename V::value_type t) { return a * (1 - t) + b * t; }
template<detail::FloatingVector V> constexpr V Clamp(V x, V lo, V hi) { return Min(Max(x, lo), hi); }
template<detail::FloatingVector V> constexpr V Clamp(V x, typename V::value_type lo, typename V::value_type hi) { return Clamp(x, V{lo}, V{hi}); }
template<detail::FloatingVector V> inline typename V::value_type Distance(V a, V b) { return Length(a - b); }
constexpr float Sign(float x) { return (x > 0.f) - (x < 0.f); }
constexpr float Mix(float a, float b, float t) { return a * (1.f - t) + b * t; }
constexpr float Clamp(float x, float lo, float hi) { return Min(Max(x, lo), hi); }
inline vec3 Pow(vec3 a, vec3 b) { return {__builtin_powf(a.x, b.x), __builtin_powf(a.y, b.y), __builtin_powf(a.z, b.z)}; }
inline vec3 Sqrt(vec3 a) { return {__builtin_sqrtf(a.x), __builtin_sqrtf(a.y), __builtin_sqrtf(a.z)}; }
} // namespace numeric
