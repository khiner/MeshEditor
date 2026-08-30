#pragma once

#include "numeric/quat.h"

#include <simd/matrix.h>
#include <simd/quaternion.h>

struct mat4;

struct mat3 {
    vec3 Columns[3]{};

    constexpr mat3() = default;
    constexpr explicit mat3(float diagonal) : Columns{{diagonal, 0.f, 0.f}, {0.f, diagonal, 0.f}, {0.f, 0.f, diagonal}} {}
    constexpr mat3(vec3 c0, vec3 c1, vec3 c2) : Columns{c0, c1, c2} {}
    constexpr mat3(float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22)
        : Columns{{m00, m01, m02}, {m10, m11, m12}, {m20, m21, m22}} {}
    explicit mat3(const mat4 &);

    constexpr vec3 &operator[](size_t i) { return Columns[i]; }
    constexpr const vec3 &operator[](size_t i) const { return Columns[i]; }
};

inline constexpr mat3 I3{1.f};

constexpr mat3 operator+(mat3 a, mat3 b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
constexpr mat3 operator-(mat3 a, mat3 b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
constexpr bool operator==(mat3 a, mat3 b) { return a[0] == b[0] && a[1] == b[1] && a[2] == b[2]; }
constexpr mat3 operator*(mat3 a, float b) { return {a[0] * b, a[1] * b, a[2] * b}; }
constexpr mat3 operator*(float a, mat3 b) { return b * a; }

namespace numeric::detail {
inline simd_float3x3 ToSimd(mat3 m) {
    return simd_matrix(simd_make_float3(m[0].x, m[0].y, m[0].z), simd_make_float3(m[1].x, m[1].y, m[1].z), simd_make_float3(m[2].x, m[2].y, m[2].z));
}

inline mat3 FromSimd(simd_float3x3 m) {
    return {{m.columns[0].x, m.columns[0].y, m.columns[0].z}, {m.columns[1].x, m.columns[1].y, m.columns[1].z}, {m.columns[2].x, m.columns[2].y, m.columns[2].z}};
}
} // namespace numeric::detail

inline vec3 operator*(mat3 a, vec3 b) {
    return {
        std::fma(a[0].x, b.x, a[1].x * b.y) + a[2].x * b.z,
        std::fma(a[0].y, b.x, a[1].y * b.y) + a[2].y * b.z,
        std::fma(a[0].z, b.x, a[1].z * b.y) + a[2].z * b.z,
    };
}
inline mat3 operator*(mat3 a, mat3 b) { return {a * b[0], a * b[1], a * b[2]}; }

namespace numeric {
using ::mat3;

inline mat3 Transpose(mat3 m) { return detail::FromSimd(simd_transpose(detail::ToSimd(m))); }
inline float Determinant(mat3 m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2]) + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
}
inline mat3 Inverse(mat3 m) { return detail::FromSimd(simd_inverse(detail::ToSimd(m))); }
inline mat3 ToMat3(quat q) {
    const float qxx = q.x * q.x, qyy = q.y * q.y, qzz = q.z * q.z;
    const float qxz = q.x * q.z, qxy = q.x * q.y, qyz = q.y * q.z;
    const float qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;
    return {
        {1.f - 2.f * (qyy + qzz), 2.f * (qxy + qwz), 2.f * (qxz - qwy)},
        {2.f * (qxy - qwz), 1.f - 2.f * (qxx + qzz), 2.f * (qyz + qwx)},
        {2.f * (qxz + qwy), 2.f * (qyz - qwx), 1.f - 2.f * (qxx + qyy)},
    };
}
inline quat ToQuat(mat3 m) {
    const float four_x_squared_minus_1 = m[0][0] - m[1][1] - m[2][2];
    const float four_y_squared_minus_1 = m[1][1] - m[0][0] - m[2][2];
    const float four_z_squared_minus_1 = m[2][2] - m[0][0] - m[1][1];
    const float four_w_squared_minus_1 = m[0][0] + m[1][1] + m[2][2];
    int biggest_index = 0;
    float four_biggest_squared_minus_1 = four_w_squared_minus_1;
    if (four_x_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_x_squared_minus_1;
        biggest_index = 1;
    }
    if (four_y_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_y_squared_minus_1;
        biggest_index = 2;
    }
    if (four_z_squared_minus_1 > four_biggest_squared_minus_1) {
        four_biggest_squared_minus_1 = four_z_squared_minus_1;
        biggest_index = 3;
    }
    const float biggest_value = __builtin_sqrtf(four_biggest_squared_minus_1 + 1.f) * .5f;
    const float multiplier = .25f / biggest_value;
    switch (biggest_index) {
        case 0: return {biggest_value, (m[1][2] - m[2][1]) * multiplier, (m[2][0] - m[0][2]) * multiplier, (m[0][1] - m[1][0]) * multiplier};
        case 1: return {(m[1][2] - m[2][1]) * multiplier, biggest_value, (m[0][1] + m[1][0]) * multiplier, (m[2][0] + m[0][2]) * multiplier};
        case 2: return {(m[2][0] - m[0][2]) * multiplier, (m[0][1] + m[1][0]) * multiplier, biggest_value, (m[1][2] + m[2][1]) * multiplier};
        case 3: return {(m[0][1] - m[1][0]) * multiplier, (m[2][0] + m[0][2]) * multiplier, (m[1][2] + m[2][1]) * multiplier, biggest_value};
        default: return {};
    }
}
} // namespace numeric

static_assert(sizeof(mat3) == 36);
