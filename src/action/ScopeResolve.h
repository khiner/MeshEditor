#pragma once

#include "action/Core.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"

namespace action {
// Visits the targets a scope resolves to.
// Scope::Entity targets `entity`, or `fallback` when `entity` is null.
// `active()` returns the Active target or null, and `selected(fn)` visits the selection.
template<typename A, typename S, typename F>
void ForEachScopeTarget(Scope scope, state::Entity entity, state::Entity fallback, A &&active, S &&selected, F &&fn) {
    switch (scope) {
        case Scope::Entity:
            if (const auto e = entity != state::Null ? entity : fallback; e != state::Null) fn(e);
            break;
        case Scope::Active:
            if (const auto e = active(); e != state::Null) fn(e);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta: selected(fn); break;
    }
}

// Visits the scope's object entities that hold T.
template<typename T, typename F>
void ForEachComponentTarget(state::Scene &r, Scope scope, state::Entity entity, state::Entity fallback, F &&fn) {
    ForEachScopeTarget(
        scope, entity, fallback,
        [&] {
            const auto e = FindActiveEntity(r);
            return e != state::Null && r.all_of<T>(e) ? e : state::Null;
        },
        [&](auto &&f) {
            for (const auto e : r.view<Selected>())
                if (r.all_of<T>(e)) f(e);
        },
        std::forward<F>(fn)
    );
}
} // namespace action
