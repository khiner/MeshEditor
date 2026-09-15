#pragma once

#include "FieldLimits.h"
#include "action/Updatable.h"

#include <type_traits>

// Typed constructors for Update and PatchFields.
namespace action {
// Ms... walks from the component down to the leaf field being written.
// Bounds come from the field's FieldLimits.
template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(Scope scope, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(Updatable<C>, "Add the component to action::UpdatableComponents");
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only");
    Update<F> a{scope, state::Null, state::Key<C>(), detail::FieldOffset<Ms...>(), std::move(v)};
    if constexpr (HasMin<Ms...>) a.Min = Limit<F>(FieldLimits<Ms...>::Min);
    if constexpr (HasMax<Ms...>) a.Max = Limit<F>(FieldLimits<Ms...>::Max);
    return a;
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOn(state::Entity e, detail::last_field<Ms...> v) {
    auto a = UpdateOf<Ms...>(Scope::Entity, std::move(v));
    a.Entity = e;
    return a;
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateActive(detail::last_field<Ms...> v) { return UpdateOf<Ms...>(Scope::Active, std::move(v)); }

// Offsets for several fields within one nested component value.
template<auto... Prefix, typename C, typename F, size_t N>
auto PatchFieldsOf(state::Entity e, std::array<F C::*, N> members, std::array<F, N> values) {
    using Component = detail::first_class<Prefix...>;
    std::array<uint16_t, N> offsets;
    for (size_t i = 0; i < N; ++i) offsets[i] = uint16_t(detail::FieldOffset<Prefix...>() + detail::MemPtrOffset(members[i]));
    return PatchFields<Component, F, N>{e, offsets, std::move(values)};
}
} // namespace action
