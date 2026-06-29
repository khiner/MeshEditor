#pragma once

#include "numeric/mat3.h"
#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <zpp_bits.h>

// zpp::bits can't reflect glm's vec/quat/mat, so each gets two ADL hooks: a members<N> count to stop zpp
// structured-binding their anonymous unions, plus a component-wise serialize (a matrix serializes as its vec columns).
namespace glm {
template<length_t L, typename T, qualifier Q>
auto serialize(const vec<L, T, Q> &) -> zpp::bits::members<size_t(L)>;
template<typename T, qualifier Q>
auto serialize(const qua<T, Q> &) -> zpp::bits::members<4>;
template<length_t C, length_t R, typename T, qualifier Q>
auto serialize(const mat<C, R, T, Q> &) -> zpp::bits::members<size_t(C)>;

template<length_t L, typename T, qualifier Q>
constexpr auto serialize(auto &archive, vec<L, T, Q> &v) {
    return [&]<length_t... I>(std::integer_sequence<length_t, I...>) { return archive(v[I]...); }(std::make_integer_sequence<length_t, L>{});
}
template<length_t L, typename T, qualifier Q>
constexpr auto serialize(auto &archive, const vec<L, T, Q> &v) {
    return [&]<length_t... I>(std::integer_sequence<length_t, I...>) { return archive(v[I]...); }(std::make_integer_sequence<length_t, L>{});
}
template<typename T, qualifier Q>
constexpr auto serialize(auto &archive, qua<T, Q> &q) { return archive(q.x, q.y, q.z, q.w); }
template<typename T, qualifier Q>
constexpr auto serialize(auto &archive, const qua<T, Q> &q) { return archive(q.x, q.y, q.z, q.w); }
template<length_t C, length_t R, typename T, qualifier Q>
constexpr auto serialize(auto &archive, mat<C, R, T, Q> &m) {
    return [&]<length_t... I>(std::integer_sequence<length_t, I...>) { return archive(m[I]...); }(std::make_integer_sequence<length_t, C>{});
}
template<length_t C, length_t R, typename T, qualifier Q>
constexpr auto serialize(auto &archive, const mat<C, R, T, Q> &m) {
    return [&]<length_t... I>(std::integer_sequence<length_t, I...>) { return archive(m[I]...); }(std::make_integer_sequence<length_t, C>{});
}
} // namespace glm
