#pragma once

#include <entt/core/type_info.hpp>
#include <entt/entity/fwd.hpp>

#include <bit>
#include <cstdint>

namespace action {
template<typename T>
struct Update {
    entt::entity Entity;
    entt::id_type ComponentType;
    uint16_t Offset;
    T Value;
};

namespace detail {
// A non-virtual data-member pointer's bit pattern is the byte offset.
template<typename P>
constexpr std::ptrdiff_t MemPtrOffset(P p) {
    static_assert(sizeof(P) == sizeof(std::ptrdiff_t));
    return std::bit_cast<std::ptrdiff_t>(p);
}

template<typename Component, typename Field>
constexpr Update<Field> MakeUpdate(entt::entity e, std::ptrdiff_t offset, Field value) {
    static_assert(std::is_trivially_copyable_v<Field>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    return {e, entt::type_hash<Component>::value(), uint16_t(offset), std::move(value)};
}
} // namespace detail

template<typename Component, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Field Component::*m, Field v) {
    return detail::MakeUpdate<Component>(e, detail::MemPtrOffset(m), std::move(v));
}
template<typename Component, typename Outer, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Outer Component::*o, Field Outer::*i, Field v) {
    return detail::MakeUpdate<Component>(e, detail::MemPtrOffset(o) + detail::MemPtrOffset(i), std::move(v));
}
template<typename Component, typename Outer, typename Middle, typename Field>
constexpr Update<Field> UpdateOf(entt::entity e, Outer Component::*o, Middle Outer::*m, Field Middle::*i, Field v) {
    return detail::MakeUpdate<Component>(e, detail::MemPtrOffset(o) + detail::MemPtrOffset(m) + detail::MemPtrOffset(i), std::move(v));
}
} // namespace action
