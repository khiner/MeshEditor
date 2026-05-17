#pragma once

#include <entt/core/type_info.hpp>
#include <entt/entity/fwd.hpp>

#include <bit>
#include <cstdint>
#include <type_traits>

namespace action {
template<typename T>
struct Update {
    entt::entity Entity;
    entt::id_type ComponentType;
    uint16_t Offset;
    T Value;
};

namespace detail {
template<auto> struct member_traits;
template<class C, class F, F C::*P>
struct member_traits<P> {
    using Class = C;
    using Field = F;
};
template<auto M> using class_of = typename member_traits<M>::Class;
template<auto M> using field_of = typename member_traits<M>::Field;

// A non-virtual data-member pointer's bit pattern is the byte offset.
template<typename P>
constexpr std::ptrdiff_t MemPtrOffset(P p) {
    static_assert(sizeof(P) == sizeof(std::ptrdiff_t));
    return std::bit_cast<std::ptrdiff_t>(p);
}

template<auto M, auto...> inline constexpr auto first_v = M;
template<auto... Ms> inline constexpr auto last_v = (Ms, ...);
template<auto... Ms> using first_class = class_of<first_v<Ms...>>;
template<auto... Ms> using last_field = field_of<last_v<Ms...>>;
} // namespace detail

// Ms... walks from the component down to the leaf field being written.
template<auto... Ms>
constexpr Update<detail::last_field<Ms...>>
UpdateOf(entt::entity e, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using F = detail::last_field<Ms...>;
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    return {e, entt::type_hash<detail::first_class<Ms...>>::value(), uint16_t((detail::MemPtrOffset(Ms) + ...)), std::move(v)};
}

// Runtime-pointer escape hatch — for the rare case where a member pointer is computed at runtime
// (e.g. a generic lambda parameter, a base-to-derived cast). Prefer the NTTP form above.
template<class C, class F>
constexpr Update<F> UpdateOf(entt::entity e, F C::*m, F v) {
    static_assert(std::is_trivially_copyable_v<F>);
    return {e, entt::type_hash<C>::value(), uint16_t(detail::MemPtrOffset(m)), std::move(v)};
}
} // namespace action
