#pragma once

#include <FastFEM/Surface2Modes.h>

#include "numeric/vec2.h"

using uvec3 = fastfem::UVec3;
using vec3 = fastfem::Vec3;
using dvec3 = fastfem::DVec3;

namespace numeric {
using ::dvec3;
using ::uvec3;
using ::vec3;

constexpr vec3 Cross(vec3 a, vec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
constexpr dvec3 Cross(dvec3 a, dvec3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline dvec3 Floor(dvec3 v) { return {__builtin_floor(v.x), __builtin_floor(v.y), __builtin_floor(v.z)}; }
inline vec3 Pow(vec3 a, vec3 b) { return {__builtin_powf(a.x, b.x), __builtin_powf(a.y, b.y), __builtin_powf(a.z, b.z)}; }
inline vec3 Sqrt(vec3 a) { return {__builtin_sqrtf(a.x), __builtin_sqrtf(a.y), __builtin_sqrtf(a.z)}; }
} // namespace numeric

static_assert(sizeof(vec3) == 12);
