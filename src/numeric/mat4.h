#pragma once

#include "numeric/mat3.h"

#include <bit>

struct mat4 {
    vec4 Columns[4]{};

    constexpr mat4() = default;
    constexpr explicit mat4(float diagonal)
        : Columns{{diagonal, 0.f, 0.f, 0.f}, {0.f, diagonal, 0.f, 0.f}, {0.f, 0.f, diagonal, 0.f}, {0.f, 0.f, 0.f, diagonal}} {}
    constexpr mat4(vec4 c0, vec4 c1, vec4 c2, vec4 c3) : Columns{c0, c1, c2, c3} {}
    constexpr mat4(float m00, float m01, float m02, float m03, float m10, float m11, float m12, float m13, float m20, float m21, float m22, float m23, float m30, float m31, float m32, float m33)
        : Columns{{m00, m01, m02, m03}, {m10, m11, m12, m13}, {m20, m21, m22, m23}, {m30, m31, m32, m33}} {}
    constexpr explicit mat4(const mat3 &m)
        : Columns{{m[0].x, m[0].y, m[0].z, 0.f}, {m[1].x, m[1].y, m[1].z, 0.f}, {m[2].x, m[2].y, m[2].z, 0.f}, {0.f, 0.f, 0.f, 1.f}} {}

    constexpr vec4 &operator[](size_t i) { return Columns[i]; }
    constexpr const vec4 &operator[](size_t i) const { return Columns[i]; }
};

inline constexpr mat4 I4{1.f};

inline mat3::mat3(const mat4 &m) : Columns{{m[0].x, m[0].y, m[0].z}, {m[1].x, m[1].y, m[1].z}, {m[2].x, m[2].y, m[2].z}} {}

constexpr mat4 operator+(mat4 a, mat4 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]}; }
constexpr mat4 operator-(mat4 a, mat4 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]}; }
constexpr bool operator==(mat4 a, mat4 b) { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]; }
constexpr mat4 operator*(mat4 a, float b) { return {a[0] * b, a[1] * b, a[2] * b, a[3] * b}; }
constexpr mat4 operator*(float a, mat4 b) { return b * a; }

namespace numeric::detail {
inline simd_float4x4 ToSimd(mat4 m) { return std::bit_cast<simd_float4x4>(m); }
inline mat4 FromSimd(simd_float4x4 m) { return std::bit_cast<mat4>(m); }
} // namespace numeric::detail

inline mat4 operator*(mat4 a, mat4 b) {
    vec4 c0 = a[0] * b[0].x;
    c0 += a[1] * b[0].y;
    c0 += a[2] * b[0].z;
    c0 += a[3] * b[0].w;
    vec4 c1 = a[0] * b[1].x;
    c1 += a[1] * b[1].y;
    c1 += a[2] * b[1].z;
    c1 += a[3] * b[1].w;
    vec4 c2 = a[0] * b[2].x;
    c2 += a[1] * b[2].y;
    c2 += a[2] * b[2].z;
    c2 += a[3] * b[2].w;
    vec4 c3 = a[0] * b[3].x;
    c3 += a[1] * b[3].y;
    c3 += a[2] * b[3].z;
    c3 += a[3] * b[3].w;
    return {c0, c1, c2, c3};
}
inline vec4 operator*(mat4 a, vec4 b) {
    const vec4 p01 = a[0] * b.x + a[1] * b.y;
    const vec4 p23 = a[2] * b.z + a[3] * b.w;
    return p01 + p23;
}

namespace numeric {
using ::mat4;

inline mat4 Transpose(mat4 m) { return detail::FromSimd(simd_transpose(detail::ToSimd(m))); }
inline mat4 Inverse(mat4 m) {
    const float coefficient_00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
    const float coefficient_02 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
    const float coefficient_03 = m[1][2] * m[2][3] - m[2][2] * m[1][3];
    const float coefficient_04 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
    const float coefficient_06 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
    const float coefficient_07 = m[1][1] * m[2][3] - m[2][1] * m[1][3];
    const float coefficient_08 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
    const float coefficient_10 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
    const float coefficient_11 = m[1][1] * m[2][2] - m[2][1] * m[1][2];
    const float coefficient_12 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
    const float coefficient_14 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
    const float coefficient_15 = m[1][0] * m[2][3] - m[2][0] * m[1][3];
    const float coefficient_16 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
    const float coefficient_18 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
    const float coefficient_19 = m[1][0] * m[2][2] - m[2][0] * m[1][2];
    const float coefficient_20 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
    const float coefficient_22 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
    const float coefficient_23 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

    const vec4 factor_0{coefficient_00, coefficient_00, coefficient_02, coefficient_03};
    const vec4 factor_1{coefficient_04, coefficient_04, coefficient_06, coefficient_07};
    const vec4 factor_2{coefficient_08, coefficient_08, coefficient_10, coefficient_11};
    const vec4 factor_3{coefficient_12, coefficient_12, coefficient_14, coefficient_15};
    const vec4 factor_4{coefficient_16, coefficient_16, coefficient_18, coefficient_19};
    const vec4 factor_5{coefficient_20, coefficient_20, coefficient_22, coefficient_23};
    const vec4 vector_0{m[1][0], m[0][0], m[0][0], m[0][0]};
    const vec4 vector_1{m[1][1], m[0][1], m[0][1], m[0][1]};
    const vec4 vector_2{m[1][2], m[0][2], m[0][2], m[0][2]};
    const vec4 vector_3{m[1][3], m[0][3], m[0][3], m[0][3]};
    const vec4 inverse_0 = vector_1 * factor_0 - vector_2 * factor_1 + vector_3 * factor_2;
    const vec4 inverse_1 = vector_0 * factor_0 - vector_2 * factor_3 + vector_3 * factor_4;
    const vec4 inverse_2 = vector_0 * factor_1 - vector_1 * factor_3 + vector_3 * factor_5;
    const vec4 inverse_3 = vector_0 * factor_2 - vector_1 * factor_4 + vector_2 * factor_5;
    const mat4 inverse{
        inverse_0 * vec4{1, -1, 1, -1},
        inverse_1 * vec4{-1, 1, -1, 1},
        inverse_2 * vec4{1, -1, 1, -1},
        inverse_3 * vec4{-1, 1, -1, 1},
    };
    const vec4 row_0{inverse[0][0], inverse[1][0], inverse[2][0], inverse[3][0]};
    const vec4 dot_0 = m[0] * row_0;
    const float determinant = (dot_0.x + dot_0.y) + (dot_0.z + dot_0.w);
    return inverse * (1.f / determinant);
}
inline mat4 ToMat4(quat q) { return mat4{ToMat3(q)}; }
inline quat ToQuat(mat4 m) { return ToQuat(mat3{m}); }
inline mat4 Translate(mat4 m, vec3 t) { return m * mat4{vec4{1, 0, 0, 0}, vec4{0, 1, 0, 0}, vec4{0, 0, 1, 0}, vec4{t.x, t.y, t.z, 1}}; }
inline mat4 Scale(mat4 m, vec3 s) { return m * mat4{vec4{s.x, 0, 0, 0}, vec4{0, s.y, 0, 0}, vec4{0, 0, s.z, 0}, vec4{0, 0, 0, 1}}; }
inline mat4 Rotate(mat4 m, float angle, vec3 axis) {
    const float cosine = std::cos(angle), sine = std::sin(angle);
    axis = Normalize(axis);
    const vec3 temp = (1.f - cosine) * axis;
    mat4 rotation;
    rotation[0][0] = cosine + temp[0] * axis[0];
    rotation[0][1] = temp[0] * axis[1] + sine * axis[2];
    rotation[0][2] = temp[0] * axis[2] - sine * axis[1];
    rotation[1][0] = temp[1] * axis[0] - sine * axis[2];
    rotation[1][1] = cosine + temp[1] * axis[1];
    rotation[1][2] = temp[1] * axis[2] + sine * axis[0];
    rotation[2][0] = temp[2] * axis[0] + sine * axis[1];
    rotation[2][1] = temp[2] * axis[1] - sine * axis[0];
    rotation[2][2] = cosine + temp[2] * axis[2];
    return {
        m[0] * rotation[0][0] + m[1] * rotation[0][1] + m[2] * rotation[0][2],
        m[0] * rotation[1][0] + m[1] * rotation[1][1] + m[2] * rotation[1][2],
        m[0] * rotation[2][0] + m[1] * rotation[2][1] + m[2] * rotation[2][2],
        m[3],
    };
}
inline vec3 EulerAngles(quat q) {
    const float pitch_y = 2.f * (q.y * q.z + q.w * q.x);
    const float pitch_x = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
    const float pitch = std::abs(pitch_x) <= std::numeric_limits<float>::epsilon() && std::abs(pitch_y) <= std::numeric_limits<float>::epsilon() ? 2.f * std::atan2(q.x, q.w) : std::atan2(pitch_y, pitch_x);
    const float yaw = std::asin(Clamp(-2.f * (q.x * q.z - q.w * q.y), -1.f, 1.f));
    const float roll_y = 2.f * (q.x * q.y + q.w * q.z);
    const float roll_x = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z;
    const float roll = std::abs(roll_x) <= std::numeric_limits<float>::epsilon() && std::abs(roll_y) <= std::numeric_limits<float>::epsilon() ? 0.f : std::atan2(roll_y, roll_x);
    return {pitch, yaw, roll};
}
inline mat4 LookAt(vec3 eye, vec3 center, vec3 up) {
    const vec3 f = Normalize(center - eye), s = Normalize(Cross(f, up)), u = Cross(s, f);
    return {vec4{s.x, u.x, -f.x, 0}, vec4{s.y, u.y, -f.y, 0}, vec4{s.z, u.z, -f.z, 0}, vec4{-Dot(s, eye), -Dot(u, eye), Dot(f, eye), 1}};
}
inline mat4 PerspectiveRhZo(float fovy, float aspect, float near, float far) {
    const float tangent = __builtin_tanf(fovy / 2.f);
    return {vec4{1.f / (aspect * tangent), 0, 0, 0}, vec4{0, 1.f / tangent, 0, 0}, vec4{0, 0, far / (near - far), -1}, vec4{0, 0, -(far * near) / (far - near), 0}};
}
inline mat4 InfinitePerspectiveRhZo(float fovy, float aspect, float near) {
    const float range = __builtin_tanf(fovy / 2.f) * near;
    const float left = -range * aspect, right = range * aspect;
    const float bottom = -range, top = range;
    return {vec4{2.f * near / (right - left), 0, 0, 0}, vec4{0, 2.f * near / (top - bottom), 0, 0}, vec4{0, 0, -1, -1}, vec4{0, 0, -near, 0}};
}
inline mat4 OrthoRhZo(float left, float right, float bottom, float top, float near, float far) {
    return {vec4{2.f / (right - left), 0, 0, 0}, vec4{0, 2.f / (top - bottom), 0, 0}, vec4{0, 0, -1.f / (far - near), 0}, vec4{-(right + left) / (right - left), -(top + bottom) / (top - bottom), -near / (far - near), 1}};
}
} // namespace numeric

static_assert(sizeof(mat4) == 64);
