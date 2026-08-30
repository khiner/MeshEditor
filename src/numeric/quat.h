#pragma once

#include "numeric/vec4.h"

#include <bit>
#include <limits>
#include <numbers>

struct quat {
    float x{}, y{}, z{}, w{1.f};
    constexpr quat() = default;
    constexpr quat(float real, float imag_x, float imag_y, float imag_z) : x(imag_x), y(imag_y), z(imag_z), w(real) {}
    constexpr quat(float real, vec3 imaginary) : x(imaginary.x), y(imaginary.y), z(imaginary.z), w(real) {}
    explicit quat(vec3 euler_xyz);
    explicit constexpr quat(vec4 xyzw);
    constexpr float &operator[](size_t i) { return (&x)[i]; }
    constexpr const float &operator[](size_t i) const { return (&x)[i]; }
};
constexpr quat::quat(vec4 xyzw) { *this = std::bit_cast<quat>(xyzw); }
constexpr quat operator-(quat q) { return {-q.w, -q.x, -q.y, -q.z}; }
constexpr bool operator==(quat a, quat b) { return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w; }
constexpr quat operator+(quat a, quat b) { return {a.w + b.w, a.x + b.x, a.y + b.y, a.z + b.z}; }
constexpr quat operator-(quat a, quat b) { return {a.w - b.w, a.x - b.x, a.y - b.y, a.z - b.z}; }
constexpr quat operator*(quat q, float s) { return {q.w * s, q.x * s, q.y * s, q.z * s}; }
constexpr quat operator*(float s, quat q) { return q * s; }
constexpr quat operator*(quat a, quat b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z, a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x};
}
constexpr quat &operator*=(quat &a, quat b) { return a = a * b; }
namespace numeric {
using ::quat;
constexpr float Dot(quat a, quat b) {
    return std::fma(a.x, b.x, a.y * b.y) + std::fma(a.z, b.z, a.w * b.w);
}
inline quat Normalize(quat q) {
    const float length = __builtin_sqrtf(Dot(q, q));
    if (length <= 0.f) return {};
    return {q.w / length, q.x / length, q.y / length, q.z / length};
}
constexpr quat Conjugate(quat q) { return {q.w, -q.x, -q.y, -q.z}; }
inline quat Inverse(quat q) {
    const quat conjugated = Conjugate(q);
    const float dot_product = Dot(q, q);
    return {conjugated.w / dot_product, conjugated.x / dot_product, conjugated.y / dot_product, conjugated.z / dot_product};
}
inline vec3 Rotate(quat q, vec3 v) {
    const vec3 u{q.x, q.y, q.z};
    const vec3 uv = Cross(u, v), uuv = Cross(u, uv);
    return v + (uv * q.w + uuv) * 2.f;
}
inline quat AngleAxis(float angle, vec3 axis) {
    const float s = std::sin(angle * .5f);
    return {std::cos(angle * .5f), axis.x * s, axis.y * s, axis.z * s};
}
inline float Angle(quat q) {
    if (std::abs(q.w) > 0.877582561890372716130286068203503191f) {
        const float angle = std::asin(std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z)) * 2.f;
        return q.w < 0.f ? std::numbers::pi_v<float> * 2.f - angle : angle;
    }
    return std::acos(q.w) * 2.f;
}
inline vec3 Axis(quat q) {
    const float squared_sine = 1.f - q.w * q.w;
    if (squared_sine <= 0.f) return {0, 0, 1};
    const float inverse_sine = 1.f / std::sqrt(squared_sine);
    return {q.x * inverse_sine, q.y * inverse_sine, q.z * inverse_sine};
}
inline quat Slerp(quat a, quat b, float t) {
    float c = Dot(a, b);
    if (c < 0) {
        b = -b;
        c = -c;
    }
    if (c > 1.f - std::numeric_limits<float>::epsilon()) {
        return {Mix(a.w, b.w, t), Mix(a.x, b.x, t), Mix(a.y, b.y, t), Mix(a.z, b.z, t)};
    }
    const float angle = std::acos(c);
    const quat numerator = a * std::sin((1.f - t) * angle) + b * std::sin(t * angle);
    const float denominator = std::sin(angle);
    return {numerator.w / denominator, numerator.x / denominator, numerator.y / denominator, numerator.z / denominator};
}
inline quat Rotation(vec3 from, vec3 to) {
    const float cosine = Dot(from, to);
    if (cosine >= 1.f - std::numeric_limits<float>::epsilon()) return {};
    if (cosine < -1.f + std::numeric_limits<float>::epsilon()) {
        vec3 axis = Cross(vec3{0, 0, 1}, from);
        if (Dot(axis, axis) < std::numeric_limits<float>::epsilon()) axis = Cross(vec3{1, 0, 0}, from);
        return AngleAxis(std::numbers::pi_v<float>, Normalize(axis));
    }
    const vec3 axis = Cross(from, to);
    const float scale = std::sqrt((1.f + cosine) * 2.f), inverse_scale = 1.f / scale;
    return {scale * .5f, axis.x * inverse_scale, axis.y * inverse_scale, axis.z * inverse_scale};
}
} // namespace numeric
inline vec3 operator*(quat q, vec3 v) { return numeric::Rotate(q, v); }
inline quat::quat(vec3 e) {
    const vec3 cosine{std::cos(e.x * .5f), std::cos(e.y * .5f), std::cos(e.z * .5f)};
    const vec3 sine{std::sin(e.x * .5f), std::sin(e.y * .5f), std::sin(e.z * .5f)};
    w = cosine.x * cosine.y * cosine.z + sine.x * sine.y * sine.z;
    x = sine.x * cosine.y * cosine.z - cosine.x * sine.y * sine.z;
    y = cosine.x * sine.y * cosine.z + sine.x * cosine.y * sine.z;
    z = cosine.x * cosine.y * sine.z - sine.x * sine.y * cosine.z;
}

static_assert(sizeof(quat) == 16);
