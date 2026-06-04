#pragma once

#include "action/Core.h"
#include "action/Dispatch.h"

#include <type_traits>

// Typed constructors for Update/SetTag.
namespace action {
// Ms... walks from the component down to the leaf field being written.
template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(entt::entity e, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    RegisterUpdateable<C>();
    return {Scope::Entity, e, entt::type_hash<C>::value(), uint16_t((detail::MemPtrOffset(Ms) + ...)), std::move(v)};
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(Scope scope, detail::last_field<Ms...> v) {
    static_assert(sizeof...(Ms) > 0, "UpdateOf requires at least one member pointer");
    using C = detail::first_class<Ms...>;
    using F = detail::last_field<Ms...>;
    static_assert(std::is_trivially_copyable_v<F>, "Update<T> is for trivially-copyable fields only; use Replace<T> for complex types");
    RegisterUpdateable<C>();
    return {scope, null_entity, entt::type_hash<C>::value(), uint16_t((detail::MemPtrOffset(Ms) + ...)), std::move(v)};
}

template<auto... Ms>
Update<detail::last_field<Ms...>> UpdateOf(detail::last_field<Ms...> v) { return UpdateOf<Ms...>(Scope::Active, std::move(v)); }

// Member pointer passed as a runtime value rather than an NTTP.
template<class C, class F>
Update<F> UpdateOf(entt::entity e, F C::*m, F v) {
    static_assert(std::is_trivially_copyable_v<F>);
    RegisterUpdateable<C>();
    return {Scope::Entity, e, entt::type_hash<C>::value(), uint16_t(detail::MemPtrOffset(m)), std::move(v)};
}
template<class C, class F>
Update<F> UpdateOf(F C::*m, F v) {
    static_assert(std::is_trivially_copyable_v<F>);
    RegisterUpdateable<C>();
    return {Scope::Active, null_entity, entt::type_hash<C>::value(), uint16_t(detail::MemPtrOffset(m)), std::move(v)};
}

template<typename Tag>
SetTag SetTagOf(entt::entity e, bool present) {
    RegisterTaggable<Tag>();
    return {Scope::Entity, e, entt::type_hash<Tag>::value(), present};
}

template<typename Tag>
SetTag SetTagOf(bool present) {
    RegisterTaggable<Tag>();
    return {Scope::Active, null_entity, entt::type_hash<Tag>::value(), present};
}

template<typename T>
SetName SetNameOf(entt::entity e, std::string name) {
    RegisterNamed<T>();
    return {e, entt::type_hash<T>::value(), std::move(name)};
}
template<typename T>
CreateNamed CreateNamedOf(std::string_view prefix) {
    RegisterNamed<T>();
    return {entt::type_hash<T>::value(), std::string{prefix}};
}
} // namespace action
