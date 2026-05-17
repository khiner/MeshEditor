#pragma once

#include <entt/core/type_info.hpp>
#include <entt/entity/fwd.hpp>

namespace action {
struct SetTag {
    entt::entity Entity;
    entt::id_type TagType;
    bool Present;
};

// Like SetTag but carries no entity — the dispatcher resolves the target via FindActiveEntity(R).
// Separate type (vs an optional field on SetTag) so we don't waste bytes storing entt::null.
struct SetActiveTag {
    entt::id_type TagType;
    bool Present;
};

template<typename Tag>
constexpr SetTag SetTagOf(entt::entity e, bool present) {
    return {e, entt::type_hash<Tag>::value(), present};
}

// No-entity overload: returns SetActiveTag; dispatcher resolves via FindActiveEntity(R).
template<typename Tag>
constexpr SetActiveTag SetTagOf(bool present) {
    return {entt::type_hash<Tag>::value(), present};
}
} // namespace action
