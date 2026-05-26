#pragma once

#include "action/Core.h"

#include <entt/entity/registry.hpp>

#include <cassert>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>

// Resolves an action's component-type hash to an operation on the concrete type.
namespace action {
namespace detail {
struct ComponentPatcher {
    void (*Patch)(entt::registry &, entt::entity, uint16_t offset, const void *src, uint16_t size);
    std::string_view Name; // entt::type_name<C> — for a startup manifest / debugging
};
inline auto &PatchTable() {
    static std::unordered_map<entt::id_type, ComponentPatcher> table;
    return table;
}
using TagFn = void (*)(entt::registry &, entt::entity, bool present);
inline auto &TagTable() {
    static std::unordered_map<entt::id_type, TagFn> table;
    return table;
}

template<typename C>
void PatchComponent(entt::registry &r, entt::entity e, uint16_t offset, const void *src, uint16_t size) {
    // Field is trivially copyable (enforced in UpdateOf), so a sized copy is equivalent to assignment.
    r.patch<C>(e, [&](C &c) { std::memcpy(reinterpret_cast<std::byte *>(&c) + offset, src, size); });
}
template<typename Tag>
void SetTagPresence(entt::registry &r, entt::entity e, bool present) {
    if (present) r.emplace_or_replace<Tag>(e);
    else r.remove<Tag>(e);
}

template<typename C>
struct PatchRegistrar {
    PatchRegistrar() { PatchTable().insert_or_assign(entt::type_hash<C>::value(), ComponentPatcher{&PatchComponent<C>, entt::type_name<C>::value()}); }
};
template<typename Tag>
struct TagRegistrar {
    TagRegistrar() { TagTable().insert_or_assign(entt::type_hash<Tag>::value(), &SetTagPresence<Tag>); }
};

// Named-component dispatch: set a `.Name` field, or create an entity with an ordinal name.
using NameFn = void (*)(entt::registry &, entt::entity, const std::string &);
using CreateNamedFn = void (*)(entt::registry &, std::string_view prefix);
inline auto &NameTable() {
    static std::unordered_map<entt::id_type, NameFn> table;
    return table;
}
inline auto &CreateNamedTable() {
    static std::unordered_map<entt::id_type, CreateNamedFn> table;
    return table;
}
template<typename T>
void SetNameImpl(entt::registry &r, entt::entity e, const std::string &name) {
    r.patch<T>(e, [&](T &x) { x.Name = name; });
}
template<typename T>
void CreateNamedImpl(entt::registry &r, std::string_view prefix) {
    r.emplace<T>(r.create(), T{.Name = std::string{prefix} + ' ' + std::to_string(r.view<T>().size())});
}
template<typename T>
struct NamedRegistrar {
    NamedRegistrar() {
        const auto h = entt::type_hash<T>::value();
        NameTable().insert_or_assign(h, &SetNameImpl<T>);
        CreateNamedTable().insert_or_assign(h, &CreateNamedImpl<T>);
    }
};

// Registration runs at startup, ready before any action is applied or replayed.
template<typename C> inline const PatchRegistrar<C> patch_registrar{};
template<typename Tag> inline const TagRegistrar<Tag> tag_registrar{};
template<typename T> inline const NamedRegistrar<T> named_registrar{};
} // namespace detail

// Called by each UpdateOf/SetTagOf so a type that can be targeted is always registered for dispatch.
template<typename C> void RegisterUpdateable() { (void)&detail::patch_registrar<C>; }
template<typename Tag> void RegisterTaggable() { (void)&detail::tag_registrar<Tag>; }
template<typename T> void RegisterNamed() { (void)&detail::named_registrar<T>; }

template<typename Field>
void ApplyUpdate(entt::registry &r, entt::entity e, entt::id_type component_type, uint16_t offset, const Field &value) {
    auto it = detail::PatchTable().find(component_type);
    assert(it != detail::PatchTable().end() && "Update target component is not registered for dispatch");
    it->second.Patch(r, e, offset, &value, sizeof(Field));
}

// An Update<Field> with no explicit entity (Entity == null) targets the viewport.
template<typename Field>
void ApplyUpdate(entt::registry &r, entt::entity viewport, const Update<Field> &a) {
    ApplyUpdate(r, a.Entity != entt::null ? a.Entity : viewport, a.ComponentType, a.Offset, a.Value);
}

inline void ApplyTag(entt::registry &r, entt::entity e, entt::id_type tag_type, bool present) {
    auto it = detail::TagTable().find(tag_type);
    assert(it != detail::TagTable().end() && "Tag type is not registered for dispatch");
    it->second(r, e, present);
}

inline void ApplySetName(entt::registry &r, entt::id_type type, entt::entity e, const std::string &name) {
    auto it = detail::NameTable().find(type);
    assert(it != detail::NameTable().end() && "SetName target component is not registered for dispatch");
    it->second(r, e, name);
}
inline void ApplyCreateNamed(entt::registry &r, entt::id_type type, std::string_view prefix) {
    auto it = detail::CreateNamedTable().find(type);
    assert(it != detail::CreateNamedTable().end() && "CreateNamed target component is not registered for dispatch");
    it->second(r, prefix);
}
} // namespace action
