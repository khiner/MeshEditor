#pragma once

#include "numeric/vec2.h"

struct uvec3 {
    using value_type = uint32_t;
    static constexpr size_t ComponentCount = 3;
    uint32_t x{}, y{}, z{};
    constexpr uvec3() = default;
    constexpr uvec3(uint32_t v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr uvec3(X x, Y y, Z z) : x(uint32_t(x)), y(uint32_t(y)), z(uint32_t(z)) {}
    constexpr uint32_t &operator[](size_t i) { return (&x)[i]; }
    constexpr const uint32_t &operator[](size_t i) const { return (&x)[i]; }
};

struct vec3 {
    using value_type = float;
    static constexpr size_t ComponentCount = 3;
    float x{}, y{}, z{};
    constexpr vec3() = default;
    constexpr vec3(float v) : x(v), y(v), z(v) {}
    template<typename X, typename Y, typename Z> constexpr vec3(X x, Y y, Z z) : x(float(x)), y(float(y)), z(float(z)) {}
    constexpr vec3(vec2 xy, float z) : x(xy.x), y(xy.y), z(z) {}
    template<typename V> constexpr explicit vec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
};

struct dvec3 {
    using value_type = double;
    static constexpr size_t ComponentCount = 3;
    double x{}, y{}, z{};
    constexpr dvec3() = default;
    constexpr dvec3(double v) : x(v), y(v), z(v) {}
    constexpr dvec3(double x, double y, double z) : x(x), y(y), z(z) {}
    template<typename V> constexpr explicit dvec3(const V &v)
        requires requires { v.x; v.y; v.z; }
        : x(v.x), y(v.y), z(v.z) {}
    constexpr double &operator[](size_t i) { return (&x)[i]; }
    constexpr const double &operator[](size_t i) const { return (&x)[i]; }
};

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
