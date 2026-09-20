#pragma once

#include "action/Updatable.h"

#include <type_traits>

// Typed constructors for Update and PatchFields.
namespace action {
// Ms... walks from the component down to the leaf field being written.
template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(Target target, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(Updatable<C>, "Add the component to action::UpdatableComponents");
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only");
    return {std::move(target), state::Key<C>(), detail::FieldOffset<Ms...>(), std::move(v)};
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateActive(detail::last_field<Ms...> v) { return UpdateOf<Ms...>(OnActive{}, std::move(v)); }

// Offsets for several fields within one nested component value.
template<auto... Prefix, typename C, typename F, size_t N>
auto PatchFieldsOf(std::array<F C::*, N> members, std::array<F, N> values) {
    using Component = detail::first_class<Prefix...>;
    std::array<uint16_t, N> offsets;
    for (size_t i = 0; i < N; ++i) offsets[i] = uint16_t(detail::FieldOffset<Prefix...>() + detail::MemPtrOffset(members[i]));
    return PatchFields<Component, F, N>{offsets, std::move(values)};
}
} // namespace action
