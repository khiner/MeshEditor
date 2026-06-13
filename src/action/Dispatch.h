#pragma once

#include "FieldLimits.h"
#include "action/Core.h"

#include <entt/entity/registry.hpp>
#include <glm/common.hpp>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

// Resolves an action's component-type hash to an operation on the concrete type.
namespace action {
namespace detail {
struct ComponentPatcher {
    void (*Patch)(entt::registry &, entt::entity, uint16_t offset, const void *src, uint16_t size);
    void (*Read)(const entt::registry &, entt::entity, uint16_t offset, void *dst, uint16_t size); // for SelectedDelta baselines
    bool (*Has)(const entt::registry &, entt::entity); // filters Active/Selected targets to those carrying C
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
template<typename C>
void ReadComponent(const entt::registry &r, entt::entity e, uint16_t offset, void *dst, uint16_t size) {
    std::memcpy(dst, reinterpret_cast<const std::byte *>(&r.get<const C>(e)) + offset, size);
}
template<typename C>
bool HasComponent(const entt::registry &r, entt::entity e) { return r.all_of<C>(e); }
template<typename Tag>
void SetTagPresence(entt::registry &r, entt::entity e, bool present) {
    if (present) r.emplace_or_replace<Tag>(e);
    else r.remove<Tag>(e);
}

template<typename C>
struct PatchRegistrar {
    PatchRegistrar() { PatchTable().insert_or_assign(entt::type_hash<C>::value(), ComponentPatcher{&PatchComponent<C>, &ReadComponent<C>, &HasComponent<C>, entt::type_name<C>::value()}); }
};
template<typename Tag>
struct TagRegistrar {
    TagRegistrar() { TagTable().insert_or_assign(entt::type_hash<Tag>::value(), &SetTagPresence<Tag>); }
};

// Field-value clamping, keyed by the same (component, field-offset) identity Update carries.
inline uint64_t LimitsKey(entt::id_type comp, uint16_t offset) { return (uint64_t(comp) << 16) | offset; }
inline auto &LimitsTable() {
    static std::unordered_map<uint64_t, void (*)(void *)> table;
    return table;
}
// Clamps a value to its field's FieldLimits (componentwise for vecs).
template<auto... Ms>
void ClampField(void *value) {
    using F = last_field<Ms...>;
    using L = FieldLimits<Ms...>;
    F &v = *static_cast<F *>(value);
    if constexpr (HasMin<Ms...>) v = glm::max(v, F(L::Min));
    if constexpr (HasMax<Ms...>) v = glm::min(v, F(L::Max));
}
template<auto... Ms>
struct LimitsRegistrar {
    LimitsRegistrar() {
        LimitsTable().insert_or_assign(LimitsKey(entt::type_hash<first_class<Ms...>>::value(), uint16_t((MemPtrOffset(Ms) + ...))), &ClampField<Ms...>);
    }
};
template<auto... Ms> inline const LimitsRegistrar<Ms...> limits_registrar{};

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
// Called by UpdateOf for a field that declares FieldLimits, so the Apply path can clamp it.
template<auto... Ms> void RegisterLimits() { (void)&detail::limits_registrar<Ms...>; }

// Clamp `value` in place to its field's FieldLimits. A no-op for unbounded fields.
inline void MaybeClamp(entt::id_type comp, uint16_t offset, void *value) {
    if (const auto it = detail::LimitsTable().find(detail::LimitsKey(comp, offset)); it != detail::LimitsTable().end()) it->second(value);
}

template<typename Field>
void ApplyUpdate(entt::registry &r, entt::entity e, entt::id_type component_type, uint16_t offset, const Field &value) {
    auto it = detail::PatchTable().find(component_type);
    assert(it != detail::PatchTable().end() && "Update target component is not registered for dispatch");
    it->second.Patch(r, e, offset, &value, sizeof(Field));
}

// Resolve `scope` to its targets and patch each (Active/Selected hit only entities carrying the component).
void ApplyUpdateScoped(entt::registry &, entt::entity viewport, Scope, entt::entity, entt::id_type component_type, uint16_t offset, const void *value, uint16_t size);
void ApplyTagScoped(entt::registry &, entt::entity viewport, Scope, entt::entity, entt::id_type tag_type, bool present);
void ForEachSelectedWith(entt::registry &, entt::id_type component_type, const std::function<void(entt::entity)> &fn);

// Fields that support SelectedDelta (numeric drag).
template<typename Field>
inline constexpr bool DeltaField = std::same_as<Field, float> || std::same_as<Field, double> || std::same_as<Field, vec2> || std::same_as<Field, vec3> || std::same_as<Field, vec4>;

// The DragFieldStart snapshot if present, else the current value (read via `p`), which it snapshots keyed by comp + offset.
template<typename Field>
Field FieldGestureStart(entt::registry &r, entt::entity e, const detail::ComponentPatcher &p, entt::id_type comp, uint16_t offset) {
    static_assert(sizeof(Field) <= sizeof(DragFieldStart::Bytes));
    Field start;
    if (const auto *snap = r.try_get<DragFieldStart>(e); snap && snap->Comp == comp && snap->Offset == offset) {
        std::memcpy(&start, snap->Bytes.data(), sizeof(Field));
        return start;
    }
    p.Read(r, e, offset, &start, sizeof(Field));
    DragFieldStart s{comp, offset, uint16_t(sizeof(Field)), {}};
    std::memcpy(s.Bytes.data(), &start, sizeof(Field));
    r.emplace_or_replace<DragFieldStart>(e, s);
    return start;
}

template<typename Field>
void ApplyUpdate(entt::registry &r, entt::entity viewport, const Update<Field> &a) {
    if constexpr (DeltaField<Field>) {
        if (a.Scope == Scope::SelectedDelta) {
            const auto it = detail::PatchTable().find(a.ComponentType);
            assert(it != detail::PatchTable().end() && "SelectedDelta target component is not registered for dispatch");
            const auto &p = it->second;
            ForEachSelectedWith(r, a.ComponentType, [&](entt::entity e) {
                Field result = FieldGestureStart<Field>(r, e, p, a.ComponentType, a.Offset) + a.Value;
                MaybeClamp(a.ComponentType, a.Offset, &result);
                p.Patch(r, e, a.Offset, &result, sizeof(Field));
            });
            return;
        }
    }
    Field value = a.Value;
    MaybeClamp(a.ComponentType, a.Offset, &value);
    ApplyUpdateScoped(r, viewport, a.Scope, a.Entity, a.ComponentType, a.Offset, &value, sizeof(Field));
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
