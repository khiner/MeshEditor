#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

struct uvec2 {
    using value_type = uint32_t;
    static constexpr size_t ComponentCount = 2;
    uint32_t x{}, y{};
    constexpr uvec2() = default;
    constexpr uvec2(uint32_t v) : x(v), y(v) {}
    template<typename X, typename Y> constexpr uvec2(X x, Y y) : x(uint32_t(x)), y(uint32_t(y)) {}
    constexpr uint32_t &operator[](size_t i) { return (&x)[i]; }
    constexpr const uint32_t &operator[](size_t i) const { return (&x)[i]; }
};

struct vec2 {
    using value_type = float;
    static constexpr size_t ComponentCount = 2;
    float x{}, y{};
    constexpr vec2() = default;
    constexpr vec2(float v) : x(v), y(v) {}
    template<typename X, typename Y> constexpr vec2(X x, Y y) : x(float(x)), y(float(y)) {}
    constexpr vec2(const uvec2 &v) : x(float(v.x)), y(float(v.y)) {}
    template<typename V> constexpr vec2(const V &v)
        requires requires { v.x; v.y; }
        : x(v.x), y(v.y) {}
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
};

namespace numeric::detail {
template<typename V>
concept Vector = requires(V v) {
    typename V::value_type;
    { V::ComponentCount } -> std::convertible_to<size_t>;
    { v[0] } -> std::same_as<typename V::value_type &>;
};
template<typename V>
concept FloatingVector = Vector<V> && std::floating_point<typename V::value_type>;
} // namespace numeric::detail

template<numeric::detail::Vector V> constexpr V operator+(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] += b[i];
    return a;
}
template<numeric::detail::Vector V> constexpr V operator-(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] -= b[i];
    return a;
}
template<numeric::detail::Vector V> constexpr V operator-(V v) {
    for (size_t i = 0; i < V::ComponentCount; ++i) v[i] = -v[i];
    return v;
}
template<numeric::detail::Vector V> constexpr V operator*(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] *= b[i];
    return a;
}
template<numeric::detail::Vector V> constexpr V operator/(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] /= b[i];
    return a;
}
template<numeric::detail::Vector V> constexpr V operator+(V v, typename V::value_type s) { return v + V{s}; }
template<numeric::detail::Vector V> constexpr V operator+(typename V::value_type s, V v) { return V{s} + v; }
template<numeric::detail::Vector V> constexpr V operator-(V v, typename V::value_type s) { return v - V{s}; }
template<numeric::detail::Vector V> constexpr V operator-(typename V::value_type s, V v) { return V{s} - v; }
template<numeric::detail::Vector V> constexpr V operator*(V v, typename V::value_type s) {
    for (size_t i = 0; i < V::ComponentCount; ++i) v[i] *= s;
    return v;
}
template<numeric::detail::Vector V> constexpr V operator*(typename V::value_type s, V v) { return v * s; }
template<numeric::detail::Vector V> constexpr V operator/(V v, typename V::value_type s) {
    for (size_t i = 0; i < V::ComponentCount; ++i) v[i] /= s;
    return v;
}
template<numeric::detail::Vector V> constexpr V operator/(typename V::value_type s, V v) { return V{s} / v; }
template<numeric::detail::Vector V> constexpr V &operator+=(V &a, V b) { return a = a + b; }
template<numeric::detail::Vector V> constexpr V &operator-=(V &a, V b) { return a = a - b; }
template<numeric::detail::Vector V> constexpr V &operator*=(V &a, V b) { return a = a * b; }
template<numeric::detail::Vector V> constexpr V &operator/=(V &a, V b) { return a = a / b; }
template<numeric::detail::Vector V> constexpr V &operator+=(V &v, typename V::value_type s) { return v = v + s; }
template<numeric::detail::Vector V> constexpr V &operator-=(V &v, typename V::value_type s) { return v = v - s; }
template<numeric::detail::Vector V> constexpr V &operator*=(V &v, typename V::value_type s) { return v = v * s; }
template<numeric::detail::Vector V> constexpr V &operator/=(V &v, typename V::value_type s) { return v = v / s; }
template<numeric::detail::Vector V> constexpr bool operator==(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i)
        if (a[i] != b[i]) return false;
    return true;
}

namespace numeric {
using ::uvec2;
using ::vec2;

template<typename T>
    requires std::is_arithmetic_v<T>
constexpr T Min(T a, T b) { return a < b ? a : b; }
template<typename T>
    requires std::is_arithmetic_v<T>
constexpr T Max(T a, T b) { return a > b ? a : b; }
template<detail::Vector V> constexpr typename V::value_type Dot(V a, V b) {
    if constexpr (detail::FloatingVector<V>) {
        if constexpr (V::ComponentCount == 2) return std::fma(a[0], b[0], a[1] * b[1]);
        if constexpr (V::ComponentCount == 3) return std::fma(a[0], b[0], std::fma(a[1], b[1], a[2] * b[2]));
        if constexpr (V::ComponentCount == 4) return std::fma(a[0], b[0], a[1] * b[1]) + std::fma(a[2], b[2], a[3] * b[3]);
    }
    const V products = a * b;
    typename V::value_type result{};
    for (size_t i = 0; i < V::ComponentCount; ++i) result += products[i];
    return result;
}
template<detail::FloatingVector V> inline typename V::value_type Length(V v) {
    if constexpr (std::same_as<typename V::value_type, float>) return __builtin_sqrtf(Dot(v, v));
    else return __builtin_sqrt(Dot(v, v));
}
template<detail::FloatingVector V> inline V Normalize(V v) { return v / Length(v); }
template<detail::Vector V> constexpr V Min(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] = Min(a[i], b[i]);
    return a;
}
template<detail::Vector V> constexpr V Max(V a, V b) {
    for (size_t i = 0; i < V::ComponentCount; ++i) a[i] = Max(a[i], b[i]);
    return a;
}
template<detail::Vector V> constexpr V Min(V a, typename V::value_type b) { return Min(a, V{b}); }
template<detail::Vector V> constexpr V Max(V a, typename V::value_type b) { return Max(a, V{b}); }
template<detail::FloatingVector V> constexpr V Abs(V v) {
    for (size_t i = 0; i < V::ComponentCount; ++i)
        if (v[i] < 0) v[i] = -v[i];
    return v;
}
template<detail::FloatingVector V> constexpr V Mix(V a, V b, typename V::value_type t) { return a * (1 - t) + b * t; }
template<detail::FloatingVector V> constexpr V Clamp(V x, V lo, V hi) { return Min(Max(x, lo), hi); }
template<detail::FloatingVector V> constexpr V Clamp(V x, typename V::value_type lo, typename V::value_type hi) { return Clamp(x, V{lo}, V{hi}); }
template<detail::FloatingVector V> inline typename V::value_type Distance(V a, V b) { return Length(a - b); }
template<detail::FloatingVector V> constexpr typename V::value_type Distance2(V a, V b) { return Dot(a - b, a - b); }

constexpr float Sign(float x) { return (x > 0.f) - (x < 0.f); }
constexpr float Mix(float a, float b, float t) { return a * (1.f - t) + b * t; }
constexpr float Clamp(float x, float lo, float hi) { return Min(Max(x, lo), hi); }
} // namespace numeric

static_assert(sizeof(vec2) == 8);
