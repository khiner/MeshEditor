#pragma once

#include "numeric/mat4.h"
#include "numeric/quat.h"

#include <bit>
#include <simd/matrix.h>

namespace numeric::detail {
inline simd_float3x3 ToSimd(mat3 m) {
    return simd_matrix(simd_make_float3(m[0].x, m[0].y, m[0].z), simd_make_float3(m[1].x, m[1].y, m[1].z), simd_make_float3(m[2].x, m[2].y, m[2].z));
}

inline mat3 FromSimd(simd_float3x3 m) {
    return {{m.columns[0].x, m.columns[0].y, m.columns[0].z}, {m.columns[1].x, m.columns[1].y, m.columns[1].z}, {m.columns[2].x, m.columns[2].y, m.columns[2].z}};
}
inline simd_float4x4 ToSimd(mat4 m) { return std::bit_cast<simd_float4x4>(m); }
inline mat4 FromSimd(simd_float4x4 m) { return std::bit_cast<mat4>(m); }
} // namespace numeric::detail

inline vec3 operator*(mat3 a, vec3 b) {
    return {
        std::fma(a[0].x, b.x, a[1].x * b.y) + a[2].x * b.z,
        std::fma(a[0].y, b.x, a[1].y * b.y) + a[2].y * b.z,
        std::fma(a[0].z, b.x, a[1].z * b.y) + a[2].z * b.z,
    };
}
inline mat3 operator*(mat3 a, mat3 b) { return {a * b[0], a * b[1], a * b[2]}; }

inline mat4 operator*(mat4 a, mat4 b) { return numeric::detail::FromSimd(simd_mul(numeric::detail::ToSimd(a), numeric::detail::ToSimd(b))); }
inline vec4 operator*(mat4 a, vec4 b) {
    return std::bit_cast<vec4>(simd_mul(numeric::detail::ToSimd(a), std::bit_cast<simd_float4>(b)));
}

namespace numeric {
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
inline mat4 LookAt(vec3 eye, vec3 center, vec3 up) {
    const vec3 f = Normalize(center - eye), s = Normalize(Cross(f, up)), u = Cross(s, f);
    return {vec4{s.x, u.x, -f.x, 0}, vec4{s.y, u.y, -f.y, 0}, vec4{s.z, u.z, -f.z, 0}, vec4{-Dot(s, eye), -Dot(u, eye), Dot(f, eye), 1}};
}
inline mat4 PerspectiveRhZo(float fovy, float aspect, float near, float far) {
    const float tangent = __builtin_tanf(fovy / 2.f);
    return {vec4{1.f / (aspect * tangent), 0, 0, 0}, vec4{0, 1.f / tangent, 0, 0}, vec4{0, 0, far / (near - far), -1}, vec4{0, 0, -(far * near) / (far - near), 0}};
}
inline mat4 InfinitePerspectiveRhZo(float fovy, float aspect, float near) {
    const float tangent = 1.f / __builtin_tanf(fovy * .5f);
    return {vec4{tangent / aspect, 0, 0, 0}, vec4{0, tangent, 0, 0}, vec4{0, 0, -1, -1}, vec4{0, 0, -near, 0}};
}
inline mat4 OrthoRhZo(float left, float right, float bottom, float top, float near, float far) {
    return {vec4{2.f / (right - left), 0, 0, 0}, vec4{0, 2.f / (top - bottom), 0, 0}, vec4{0, 0, -1.f / (far - near), 0}, vec4{-(right + left) / (right - left), -(top + bottom) / (top - bottom), -near / (far - near), 1}};
}
} // namespace numeric
