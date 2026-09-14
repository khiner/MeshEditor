#pragma once

#include "action/Core.h"
#include "action/Dispatch.h"

#include <type_traits>

// Typed constructors for Update/SetTag.
namespace action {
// Ms... walks from the component down to the leaf field being written.
template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(Scope scope, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    RegisterUpdateable<C>();
    if constexpr (HasLimits<Ms...>) RegisterLimits<Ms...>();
    return {scope, null_entity, state::Type<C>(), detail::FieldOffset<Ms...>(), std::move(v)};
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(state::Entity e, detail::last_field<Ms...> v) {
    auto a = UpdateOf<Ms...>(Scope::Entity, std::move(v));
    a.Entity = e;
    return a;
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(detail::last_field<Ms...> v) { return UpdateOf<Ms...>(Scope::Active, std::move(v)); }

// Offsets for several fields within one nested component value.
template<auto... Prefix, typename C, typename F, size_t N>
auto PatchFieldsOf(state::Entity e, std::array<F C::*, N> members, std::array<F, N> values) {
    using Component = detail::first_class<Prefix...>;
    std::array<uint16_t, N> offsets;
    for (size_t i = 0; i < N; ++i) offsets[i] = uint16_t(detail::FieldOffset<Prefix...>() + detail::MemPtrOffset(members[i]));
    return PatchFields<Component, F, N>{e, offsets, std::move(values)};
}

// Member pointer passed as a runtime value rather than an NTTP.
template<typename C, typename F>
Update<F> UpdateOf(state::Entity e, F C::*m, F v) {
    static_assert(std::is_trivially_copyable_v<F>);
    RegisterUpdateable<C>();
    return {Scope::Entity, e, state::Type<C>(), uint16_t(detail::MemPtrOffset(m)), std::move(v)};
}
template<typename C, typename F>
Update<F> UpdateOf(F C::*m, F v) {
    static_assert(std::is_trivially_copyable_v<F>);
    RegisterUpdateable<C>();
    return {Scope::Active, null_entity, state::Type<C>(), uint16_t(detail::MemPtrOffset(m)), std::move(v)};
}

template<typename Tag>
SetTag SetTagOf(state::Entity e, bool present) {
    RegisterTaggable<Tag>();
    return {Scope::Entity, e, state::Type<Tag>(), present};
}

template<typename Tag>
SetTag SetTagOf(bool present) {
    RegisterTaggable<Tag>();
    return {Scope::Active, null_entity, state::Type<Tag>(), present};
}

template<typename T>
SetName SetNameOf(state::Entity e, std::string name) {
    RegisterNamed<T>();
    return {e, state::Type<T>(), std::move(name)};
}
template<typename T>
CreateNamed CreateNamedOf(std::string_view prefix) {
    RegisterNamed<T>();
    return {state::Type<T>(), std::string{prefix}};
}
} // namespace action
