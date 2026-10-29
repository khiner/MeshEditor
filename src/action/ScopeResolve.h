#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "scene/Entity.h"
#include "scene/HierarchyInstance.h"
#include "selection/Selection.h"
#include "state/Registry.h"

namespace action {
// Visits the entities a target resolves to.
// `active()` returns the active target or null, and `selected(fn)` visits the selection for both selection targets.
void ForEachTarget(const Target &target, auto &&active, auto &&selected, auto &&fn) {
    std::visit(
        overloaded{
            [&](OnActive) {
                if (const auto e = active(); e != state::Null) fn(e);
            },
            [&](state::Entity e) { fn(e); },
            [&](auto) { selected(fn); },
        },
        target
    );
}

template<typename T> state::Entity ActiveWith(const state::Registry &r) {
    const auto e = FindActiveEntity(r);
    return e != state::Null && r.all_of<T>(e) ? e : state::Null;
}
template<typename T> void ForEachSelectedWith(const state::Registry &r, auto &&fn) {
    for (const auto e : SelectedIn(r)) {
        if (r.all_of<T>(e)) fn(e);
    }
}
// Visits each entity holding T that the target resolves to.
template<typename T, typename F>
void ForEachComponentTarget(state::Registry &r, const Target &target, F &&fn) {
    ForEachTarget(target, [&] { return ActiveWith<T>(r); }, [&](auto &&f) { ForEachSelectedWith<T>(r, f); }, [&](state::Entity e) { fn(HierarchySource(r, e)); });
}
} // namespace action
