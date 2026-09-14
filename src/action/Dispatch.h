#pragma once

#include "FieldLimits.h"
#include "action/Core.h"
#include "state/Scene.h"

#include <cassert>
#include <concepts>
#include <functional>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>

// Dynamic commands dispatch once through the fixed component catalog.
namespace action {
namespace detail {
using PatchFn = void (*)(state::Scene &, state::Entity, uint16_t offset, const void *src, uint16_t size);
inline auto &PatchTable() {
    static std::array<PatchFn, state::SchemaSize> table{};
    return table;
}
using TagFn = void (*)(state::Scene &, state::Entity, bool present);
inline auto &TagTable() {
    static std::array<TagFn, state::SchemaSize> table{};
    return table;
}

template<typename C>
void PatchComponent(state::Scene &r, state::Entity e, uint16_t offset, const void *src, uint16_t size) {
    // Field is trivially copyable (enforced in UpdateOf), so a sized copy is equivalent to assignment.
    r.patch<C>(e, [&](C &c) { std::memcpy(reinterpret_cast<std::byte *>(&c) + offset, src, size); });
}
template<typename Tag>
void SetTagPresence(state::Scene &r, state::Entity e, bool present) {
    if (present) r.emplace_or_replace<Tag>(e);
    else r.remove<Tag>(e);
}

template<typename C>
struct PatchRegistrar {
    PatchRegistrar() { PatchTable()[state::Type<C>()] = &PatchComponent<C>; }
};
template<typename Tag>
struct TagRegistrar {
    TagRegistrar() { TagTable()[state::Type<Tag>()] = &SetTagPresence<Tag>; }
};

// Field-value clamping, keyed by (component, field-offset, field-size). Size separates a whole-field clamp from a per-component clamp at the same offset.
inline uint64_t LimitsKey(state::TypeId comp, uint16_t offset, uint16_t size) { return (uint64_t(comp) << 32) | (uint64_t(offset) << 16) | size; }
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
    if constexpr (HasMin<Ms...>) v = numeric::Max(v, F(L::Min));
    if constexpr (HasMax<Ms...>) v = numeric::Min(v, F(L::Max));
}
// Clamps one scalar component of a vec field to the field's (componentwise) FieldLimits.
template<auto... Ms>
void ClampComponent(void *value) {
    using E = typename last_field<Ms...>::value_type;
    using L = FieldLimits<Ms...>;
    E &v = *static_cast<E *>(value);
    if constexpr (HasMin<Ms...>) v = numeric::Max(v, E(L::Min));
    if constexpr (HasMax<Ms...>) v = numeric::Min(v, E(L::Max));
}
template<auto... Ms>
struct LimitsRegistrar {
    LimitsRegistrar() {
        using F = last_field<Ms...>;
        const auto comp = state::Type<first_class<Ms...>>();
        const auto base = FieldOffset<Ms...>();
        LimitsTable().insert_or_assign(LimitsKey(comp, base, sizeof(F)), &ClampField<Ms...>);
        // A vec field can also be patched one component at a time, so register the same bounds per component.
        if constexpr (requires { F::ComponentCount; }) {
            using E = typename F::value_type;
            for (size_t i = 0; i < F::ComponentCount; ++i)
                LimitsTable().insert_or_assign(LimitsKey(comp, uint16_t(base + i * sizeof(E)), sizeof(E)), &ClampComponent<Ms...>);
        }
    }
};
template<auto... Ms> inline const LimitsRegistrar<Ms...> limits_registrar{};

// Named-component dispatch: set a `.Name` field, or create an entity with an ordinal name.
using NameFn = void (*)(state::Scene &, state::Entity, const std::string &);
using CreateNamedFn = void (*)(state::Scene &, std::string_view prefix);
inline auto &NameTable() {
    static std::array<NameFn, state::SchemaSize> table{};
    return table;
}
inline auto &CreateNamedTable() {
    static std::array<CreateNamedFn, state::SchemaSize> table{};
    return table;
}
template<typename T>
void SetNameImpl(state::Scene &r, state::Entity e, const std::string &name) {
    r.patch<T>(e, [&](T &x) { x.Name = name; });
}
template<typename T>
void CreateNamedImpl(state::Scene &r, std::string_view prefix) {
    r.emplace<T>(r.create(), T{.Name = std::string{prefix} + ' ' + std::to_string(r.view<T>().size())});
}
template<typename T>
struct NamedRegistrar {
    NamedRegistrar() {
        const auto h = state::Type<T>();
        NameTable()[h] = &SetNameImpl<T>;
        CreateNamedTable()[h] = &CreateNamedImpl<T>;
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

// Clamp `value` (a `size`-byte field or component) in place to its FieldLimits. A no-op for unbounded fields.
inline void MaybeClamp(state::TypeId comp, uint16_t offset, uint16_t size, void *value) {
    if (const auto it = detail::LimitsTable().find(detail::LimitsKey(comp, offset, size)); it != detail::LimitsTable().end()) it->second(value);
}

template<typename Field>
void ApplyUpdate(state::Scene &r, state::Entity e, state::TypeId component_type, uint16_t offset, const Field &value) {
    const auto patcher = detail::PatchTable().at(component_type);
    assert(patcher);
    patcher(r, e, offset, &value, sizeof(Field));
}

// Resolves scope and patches Active or Selected targets that contain the component.
void ApplyUpdateScoped(state::Scene &, state::Entity viewport, Scope, state::Entity, state::TypeId component_type, uint16_t offset, const void *value, uint16_t size);
void ApplyTagScoped(state::Scene &, state::Entity viewport, Scope, state::Entity, state::TypeId tag_type, bool present);
void ForEachSelectedWith(state::Scene &, state::TypeId component_type, const std::function<void(state::Entity)> &);

// Arithmetic type fields that support SelectedDelta (numeric drag)
template<typename Field>
inline constexpr bool DeltaField = std::same_as<Field, float> || std::same_as<Field, double> || std::same_as<Field, vec2> || std::same_as<Field, vec3> || std::same_as<Field, vec4> || (std::integral<Field> && !std::same_as<Field, bool>);

// Cache each selected target's initial field value for the duration of a drag.
template<typename Field>
Field FieldGestureStart(state::Scene &r, state::Entity e, state::TypeId comp, uint16_t offset, auto &&read) {
    static_assert(sizeof(Field) <= sizeof(DragFieldStart::Bytes));
    Field start;
    if (const auto *snap = r.try_get<DragFieldStart>(e); snap && snap->Comp == comp && snap->Offset == offset) {
        std::memcpy(&start, snap->Bytes.data(), sizeof(Field));
        return start;
    }
    read(start);
    DragFieldStart s{comp, offset, uint16_t(sizeof(Field)), {}};
    std::memcpy(s.Bytes.data(), &start, sizeof(Field));
    r.emplace_or_replace<DragFieldStart>(e, s);
    return start;
}

template<typename Field>
void ApplyUpdate(state::Scene &r, state::Entity viewport, const Update<Field> &a) {
    if constexpr (DeltaField<Field>) {
        if (a.Scope == Scope::SelectedDelta) {
            const auto patch = detail::PatchTable().at(a.ComponentType);
            assert(patch);
            ForEachSelectedWith(r, a.ComponentType, [&](state::Entity e) {
                const Field start = FieldGestureStart<Field>(r, e, a.ComponentType, a.Offset, [&](Field &v) {
                    std::memcpy(&v, static_cast<const std::byte *>(r.storage(a.ComponentType)->value(e)) + a.Offset, sizeof(Field));
                });
                Field result;
                if constexpr (std::integral<Field>) {
                    // Accumulate in a wider signed type and clamp to the field's range so a downward delta can't wrap.
                    result = Field(std::clamp<int64_t>(int64_t(start) + int64_t(std::make_signed_t<Field>(a.Value)), int64_t(std::numeric_limits<Field>::min()), int64_t(std::numeric_limits<Field>::max())));
                } else {
                    result = start + a.Value;
                }
                MaybeClamp(a.ComponentType, a.Offset, sizeof(Field), &result);
                patch(r, e, a.Offset, &result, sizeof(Field));
            });
            return;
        }
    }
    Field value = a.Value;
    MaybeClamp(a.ComponentType, a.Offset, sizeof(Field), &value);
    ApplyUpdateScoped(r, viewport, a.Scope, a.Entity, a.ComponentType, a.Offset, &value, sizeof(Field));
}

inline void ApplyTag(state::Scene &r, state::Entity e, state::TypeId tag_type, bool present) {
    const auto apply = detail::TagTable().at(tag_type);
    assert(apply);
    apply(r, e, present);
}

inline void ApplySetName(state::Scene &r, state::TypeId type, state::Entity e, const std::string &name) {
    const auto apply = detail::NameTable().at(type);
    assert(apply);
    apply(r, e, name);
}
inline void ApplyCreateNamed(state::Scene &r, state::TypeId type, std::string_view prefix) {
    const auto apply = detail::CreateNamedTable().at(type);
    assert(apply);
    apply(r, prefix);
}
} // namespace action
