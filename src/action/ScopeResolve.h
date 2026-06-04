#pragma once

#include "action/Core.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"

#include <entt/entity/registry.hpp>

// Scope resolution for Replace<T> handlers whose component lives on the object entity (mesh-data components
// map object→mesh entity separately).
namespace action {
// `fn(entity)` for each target of `scope`. Active/Selected hit only entities already carrying T, so a copy
// never adds T to one that lacks it.
template<typename T, typename F>
void ForEachReplaceTarget(entt::registry &r, Scope scope, entt::entity entity, F &&fn) {
    switch (scope) {
        case Scope::Entity: fn(entity); break;
        case Scope::Active:
            if (const auto e = FindActiveEntity(r); e != null_entity && r.all_of<T>(e)) fn(e);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta: // whole-value Replace can't delta — copy
            for (const auto e : r.view<Selected, T>()) fn(e);
            break;
    }
}
} // namespace action
