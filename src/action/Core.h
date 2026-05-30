#pragma once

#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include <entt/core/type_info.hpp>
#include <entt/entity/fwd.hpp>

#include <string>
#include <variant>

namespace action {
template<typename T>
struct Update {
    entt::entity Entity;
    entt::id_type ComponentType;
    uint16_t Offset;
    T Value;
};

template<typename T>
struct UpdateActive {
    entt::id_type ComponentType;
    uint16_t Offset;
    T Value;
};

template<typename T>
struct Replace {
    entt::entity Entity;
    T Value;
};

template<typename T>
struct ReplaceActive {
    T Value;
};

struct DestroyEntity {
    entt::entity Entity;
};

struct SetTag {
    entt::entity Entity;
    entt::id_type TagType;
    bool Present;
};

struct SetActiveTag {
    entt::id_type TagType;
    bool Present;
};

// Set the `Name` field of the component identified by `ComponentType`.
struct SetName {
    entt::entity Entity;
    entt::id_type ComponentType;
    std::string Name;
};

// Create a new entity carrying the component identified by `ComponentType`, named "<Prefix> <ordinal>".
struct CreateNamed {
    entt::id_type ComponentType;
    std::string Prefix;
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

using Core = std::variant<
    Update<bool>, Update<uint8_t>, Update<uint32_t>, Update<float>, Update<double>,
    Update<vec3>, Update<vec4>, Update<entt::entity>,
    UpdateActive<bool>, UpdateActive<uint8_t>, UpdateActive<uint32_t>, UpdateActive<float>, UpdateActive<double>,
    UpdateActive<vec3>, UpdateActive<vec4>, UpdateActive<entt::entity>,
    SetTag, SetActiveTag, DestroyEntity>;

void Apply(entt::registry &, entt::entity viewport, const Core &);
} // namespace action
