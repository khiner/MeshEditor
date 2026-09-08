#pragma once

#include "numeric/mat3.h"
#include "numeric/vec4.h"

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

namespace numeric {
using ::mat4;
}

static_assert(sizeof(mat4) == 64);
