#pragma once

#include "Variant.h"
#include "action/Core.h"
#include "scene/Entity.h"
#include "state/Registry.h"

namespace action {
// Visits the entities a target resolves to.
// `active()` returns the active target or null, and `selected(fn)` visits the selection for both selection targets.
void ForEachTarget(const Target &target, state::Entity viewport, auto &&active, auto &&selected, auto &&fn) {
    std::visit(
        overloaded{
            [&](OnActive) {
                if (const auto e = active(); e != state::Null) fn(e);
            },
            [&](OnViewport) { fn(viewport); },
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
    for (const auto e : r.view<Selected>()) {
        if (r.all_of<T>(e)) fn(e);
    }
}
template<typename T, typename F>
void ForEachComponentTarget(state::Registry &r, const Target &target, state::Entity viewport, F &&fn) {
    ForEachTarget(target, viewport, [&] { return ActiveWith<T>(r); }, [&](auto &&f) { ForEachSelectedWith<T>(r, f); }, std::forward<F>(fn));
}
} // namespace action
