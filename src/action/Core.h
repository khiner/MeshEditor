#pragma once

#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"
#include "state/Entity.h"

#include "state/Schema.h"

#include <array>
#include <string>
#include <variant>

namespace action {
// `Selected` copies the value to each selected entity.
// `SelectedDelta` adds it as a delta to each selected entity's gesture-start value (number drags).
enum class Scope : uint8_t {
    Entity,
    Active,
    Selected,
    SelectedDelta
};

// A field's value at the start of a SelectedDelta drag, so each step (and replay) computes start + delta.
struct DragFieldStart {
    state::TypeId Comp;
    uint16_t Offset, Size;
    std::array<std::byte, 16> Bytes; // fits the widest Update field (vec4)
};

template<typename T>
struct Update {
    Scope Scope{Scope::Entity};
    state::Entity Entity{null_entity}; // Scope::Entity only. Null targets the viewport
    state::TypeId ComponentType;
    uint16_t Offset;
    T Value;
};

template<typename T>
struct Replace {
    Scope Scope{Scope::Entity};
    state::Entity Entity{null_entity}; // Scope::Entity only
    T Value;
};

// Assign authored fields together, preserving the rest of the component.
template<typename Component, typename Field, size_t N = 1>
struct PatchFields {
    state::Entity Entity;
    std::array<uint16_t, N> Offsets;
    std::array<Field, N> Values;
};

struct DestroyEntity {
    state::Entity Entity;
};

struct SetTag {
    Scope Scope{Scope::Entity};
    state::Entity Entity{null_entity}; // Scope::Entity only
    state::TypeId TagType;
    bool Present;
};

// Set the `Name` field of the component identified by `ComponentType`.
struct SetName {
    state::Entity Entity;
    state::TypeId ComponentType;
    std::string Name;
};

// Creates an entity with ComponentType and the name "<Prefix> <ordinal>".
struct CreateNamed {
    state::TypeId ComponentType;
    std::string Prefix;
};

namespace detail {
template<auto> struct member_traits;
template<typename C, typename F, F C::*P>
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

template<auto... Ms>
uint16_t FieldOffset() { return uint16_t((MemPtrOffset(Ms) + ...)); }

template<auto M, auto...> inline constexpr auto first_v = M;
template<auto... Ms> inline constexpr auto last_v = (Ms, ...);
template<auto... Ms> using first_class = class_of<first_v<Ms...>>;
template<auto... Ms> using last_field = field_of<last_v<Ms...>>;
} // namespace detail

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec2>, Update<vec3>, Update<vec4>, Update<state::Entity>,
    SetTag, DestroyEntity>;

void Apply(state::Scene &, state::Entity viewport, const Core &);
} // namespace action
