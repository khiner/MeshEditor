#pragma once

#include "action/Core.h"

#include <entt/core/type_info.hpp>
#include <entt/entity/registry.hpp>

#include <cstddef>
#include <cstdint>

// Shared machinery for applying the templated Update/Replace/Tag actions.
// Each Apply TU supplies its own `TypeList` of candidate component types so it only pulls the
// component headers it actually writes; the dispatch logic itself lives here once.
namespace action {
template<typename...> struct TypeList {};

template<typename... Ts, typename Fn>
bool DispatchByTypeHash(TypeList<Ts...>, entt::id_type hash, Fn &&fn) {
    return ((entt::type_hash<Ts>::value() == hash && (fn.template operator()<Ts>(), true)) || ...);
}

// Writes `value` at `offset` into whichever component in `Components` matches `component_type`.
template<typename Components, typename Field>
void ApplyUpdate(entt::registry &r, entt::entity e, entt::id_type component_type, uint16_t offset, const Field &value) {
    DispatchByTypeHash(Components{}, component_type, [&]<typename T> {
        r.patch<T>(e, [&](T &t) { *reinterpret_cast<Field *>(reinterpret_cast<std::byte *>(&t) + offset) = value; });
    });
}

// An Update<Field> with no explicit entity (Entity == null) targets the viewport; otherwise its own entity.
template<typename Components, typename Field>
void ApplyUpdate(entt::registry &r, entt::entity viewport, const Update<Field> &a) {
    ApplyUpdate<Components>(r, a.Entity != entt::null ? a.Entity : viewport, a.ComponentType, a.Offset, a.Value);
}

template<typename Components>
void ApplyTag(entt::registry &r, entt::entity e, entt::id_type tag_type, bool present) {
    DispatchByTypeHash(Components{}, tag_type, [&]<typename T> {
        if (present) r.emplace_or_replace<T>(e);
        else r.remove<T>(e);
    });
}
} // namespace action
