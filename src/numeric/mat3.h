#pragma once

#include "numeric/vec3.h"

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

namespace numeric {
using ::mat3;
}

static_assert(sizeof(mat3) == 36);
