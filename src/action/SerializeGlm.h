#pragma once

#include "numeric/quat.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <zpp_bits.h>

// zpp::bits can't reflect glm's vec/quat (non-aggregate, anonymous-union members) nor byte-copy them.
// Two ADL hooks in namespace glm: a `members<N>` count, so zpp's traits don't try to structured-bind
// them (a hard error for anonymous unions under C++26 binding packs), plus the component-wise serialize.
namespace glm {
template<length_t L, typename T, qualifier Q>
auto serialize(const vec<L, T, Q> &) -> zpp::bits::members<std::size_t(L)>;
template<typename T, qualifier Q>
auto serialize(const qua<T, Q> &) -> zpp::bits::members<4>;

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
} // namespace glm
