#pragma once

#include "action/Core.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"

#include <entt/entity/registry.hpp>

// Scope resolution for handlers whose component lives on the object entity (mesh-data components
// map object→mesh entity separately).
namespace action {
// `fn(entity)` for each target of `scope`. Entity resolves to `fallback` when the carried entity is
// null. Active/Selected hit only entities passing `accept` (SelectedDelta copies — actions that
// can't delta fan the same value out).
template<typename A, typename F>
void ForEachScopeTarget(entt::registry &r, Scope scope, entt::entity entity, entt::entity fallback, A &&accept, F &&fn) {
    switch (scope) {
        case Scope::Entity: fn(entity != null_entity ? entity : fallback); break;
        case Scope::Active:
            if (const auto e = FindActiveEntity(r); e != null_entity && accept(e)) fn(e);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta:
            for (const auto e : r.view<Selected>())
                if (accept(e)) fn(e);
            break;
    }
}

// `fn(entity)` for each target of `scope` already carrying T, so a copy never adds T to an entity that lacks it.
template<typename T, typename F>
void ForEachReplaceTarget(entt::registry &r, Scope scope, entt::entity entity, F &&fn) {
    ForEachScopeTarget(r, scope, entity, entity, [&](entt::entity e) { return r.all_of<T>(e); }, std::forward<F>(fn));
}
} // namespace action
