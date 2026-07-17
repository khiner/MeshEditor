#include "action/Core.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "action/ScopeResolve.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"

#include <entt/entity/registry.hpp>

namespace action {
void ApplyUpdateScoped(entt::registry &r, entt::entity viewport, Scope scope, entt::entity entity, entt::id_type component_type, uint16_t offset, const void *value, uint16_t size) {
    auto it = detail::PatchTable().find(component_type);
    assert(it != detail::PatchTable().end() && "Update target component is not registered for dispatch");
    const auto &patcher = it->second;
    ForEachScopeTarget(
        r, scope, entity, viewport,
        [&](entt::entity e) { return patcher.Has(r, e); },
        [&](entt::entity e) { patcher.Patch(r, e, offset, value, size); }
    );
}

void ForEachSelectedWith(entt::registry &r, entt::id_type component_type, const std::function<void(entt::entity)> &fn) {
    const auto it = detail::PatchTable().find(component_type);
    if (it == detail::PatchTable().end()) return;
    const auto &has = it->second.Has;
    for (const auto e : r.view<Selected>())
        if (has(r, e)) fn(e);
}

void ApplyTagScoped(entt::registry &r, entt::entity viewport, Scope scope, entt::entity entity, entt::id_type tag_type, bool present) {
    ForEachScopeTarget(r, scope, entity, viewport, [](entt::entity) { return true; }, [&](entt::entity e) { ApplyTag(r, e, tag_type, present); });
}

void Apply(entt::registry &r, entt::entity viewport, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&](const SetTag &a) { ApplyTagScoped(r, viewport, a.Scope, a.Entity, a.TagType, a.Present); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}
} // namespace action
