#include "action/Core.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"

namespace action {
void ApplyUpdateScoped(state::Scene &r, state::Entity viewport, Scope scope, state::Entity entity, state::TypeId component_type, uint16_t offset, const void *value, uint16_t size) {
    const auto patcher = detail::PatchTable().at(component_type);
    assert(patcher);
    const auto *components = r.storage(component_type);
    ForEachScopeTarget(
        r, scope, entity, viewport,
        [&](state::Entity e) { return components && components->contains(e); },
        [&](state::Entity e) { patcher(r, e, offset, value, size); }
    );
}

void ForEachSelectedWith(state::Scene &r, state::TypeId component_type, const std::function<void(state::Entity)> &fn) {
    const auto *components = r.storage(component_type);
    if (!components) return;
    for (const auto e : r.view<Selected>())
        if (components->contains(e)) fn(e);
}

void ApplyTagScoped(state::Scene &r, state::Entity viewport, Scope scope, state::Entity entity, state::TypeId tag_type, bool present) {
    ForEachScopeTarget(r, scope, entity, viewport, [](state::Entity) { return true; }, [&](state::Entity e) { ApplyTag(r, e, tag_type, present); });
}

void Apply(state::Scene &r, state::Entity viewport, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&](const SetTag &a) { ApplyTagScoped(r, viewport, a.Scope, a.Entity, state::Slot(a.TagType), a.Present); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}
} // namespace action
