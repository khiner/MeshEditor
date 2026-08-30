#pragma once

#include "numeric/vec3.h"

struct uvec4 {
    using value_type = uint32_t;
    static constexpr size_t ComponentCount = 4;
    uint32_t x{}, y{}, z{}, w{};
    constexpr uvec4() = default;
    constexpr uvec4(uint32_t v) : x(v), y(v), z(v), w(v) {}
    template<typename X, typename Y, typename Z, typename W> constexpr uvec4(X x, Y y, Z z, W w) : x(uint32_t(x)), y(uint32_t(y)), z(uint32_t(z)), w(uint32_t(w)) {}
    constexpr uint32_t &operator[](size_t i) { return (&x)[i]; }
    constexpr const uint32_t &operator[](size_t i) const { return (&x)[i]; }
};

struct vec4 {
    using value_type = float;
    static constexpr size_t ComponentCount = 4;
    float x{}, y{}, z{}, w{};
    constexpr vec4() = default;
    constexpr vec4(float v) : x(v), y(v), z(v), w(v) {}
    template<typename X, typename Y, typename Z, typename W> constexpr vec4(X x, Y y, Z z, W w) : x(float(x)), y(float(y)), z(float(z)), w(float(w)) {}
    constexpr vec4(vec3 xyz, float w) : x(xyz.x), y(xyz.y), z(xyz.z), w(w) {}
    template<typename V> constexpr explicit vec4(const V &v)
        requires requires { v.x; v.y; v.z; v.w; }
        : x(v.x), y(v.y), z(v.z), w(v.w) {}
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
};

namespace numeric {
using ::uvec4;
using ::vec4;
} // namespace numeric

static_assert(sizeof(vec4) == 16);
