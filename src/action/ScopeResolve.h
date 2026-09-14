#pragma once

#include "action/Core.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"

#include "state/Scene.h"

// Scope resolution for handlers whose component lives on the object entity (mesh-data components map object→mesh entity separately).
namespace action {
template<typename A, typename F>
void ForEachScopeTarget(state::Scene &r, Scope scope, state::Entity entity, state::Entity fallback, A &&accept, F &&fn) {
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

// Calls fn for each scope target that contains T.
template<typename T, typename F>
void ForEachReplaceTarget(state::Scene &r, Scope scope, state::Entity entity, F &&fn) {
    ForEachScopeTarget(r, scope, entity, entity, [&](state::Entity e) { return r.all_of<T>(e); }, std::forward<F>(fn));
}
} // namespace action
