#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <zpp_bits.h>

namespace numeric {
template<typename V>
constexpr auto SerializeVector(auto &archive, V &v) {
    if constexpr (std::remove_cv_t<V>::ComponentCount == 2) return archive(v.x, v.y);
    else if constexpr (std::remove_cv_t<V>::ComponentCount == 3) return archive(v.x, v.y, v.z);
    else return archive(v.x, v.y, v.z, v.w);
}

#define NUMERIC_SERIALIZE_VECTOR(Type, Count)                                                \
    auto serialize(const Type &) -> zpp::bits::members<Count>;                               \
    constexpr auto serialize(auto &archive, Type &v) { return SerializeVector(archive, v); } \
    constexpr auto serialize(auto &archive, const Type &v) { return SerializeVector(archive, v); }
NUMERIC_SERIALIZE_VECTOR(vec2, 2)
NUMERIC_SERIALIZE_VECTOR(vec3, 3)
NUMERIC_SERIALIZE_VECTOR(vec4, 4)
NUMERIC_SERIALIZE_VECTOR(uvec2, 2)
NUMERIC_SERIALIZE_VECTOR(uvec3, 3)
NUMERIC_SERIALIZE_VECTOR(uvec4, 4)
NUMERIC_SERIALIZE_VECTOR(dvec3, 3)
#undef NUMERIC_SERIALIZE_VECTOR

auto serialize(const quat &) -> zpp::bits::members<4>;
constexpr auto serialize(auto &archive, quat &q) { return archive(q.x, q.y, q.z, q.w); }
constexpr auto serialize(auto &archive, const quat &q) { return archive(q.x, q.y, q.z, q.w); }
auto serialize(const mat3 &) -> zpp::bits::members<3>;
constexpr auto serialize(auto &archive, mat3 &m) { return archive(m[0], m[1], m[2]); }
constexpr auto serialize(auto &archive, const mat3 &m) { return archive(m[0], m[1], m[2]); }
auto serialize(const mat4 &) -> zpp::bits::members<4>;
constexpr auto serialize(auto &archive, mat4 &m) { return archive(m[0], m[1], m[2], m[3]); }
constexpr auto serialize(auto &archive, const mat4 &m) { return archive(m[0], m[1], m[2], m[3]); }
} // namespace numeric
