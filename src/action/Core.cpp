#include "action/Core.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"

#include <entt/entity/registry.hpp>

namespace action {
void ApplyUpdateScoped(entt::registry &r, entt::entity viewport, Scope scope, entt::entity entity, entt::id_type component_type, uint16_t offset, const void *value, uint16_t size) {
    auto it = detail::PatchTable().find(component_type);
    assert(it != detail::PatchTable().end() && "Update target component is not registered for dispatch");
    const auto &patcher = it->second;
    switch (scope) {
        case Scope::Entity: patcher.Patch(r, entity != null_entity ? entity : viewport, offset, value, size); break;
        case Scope::Active:
            if (const auto e = FindActiveEntity(r); e != null_entity && patcher.Has(r, e)) patcher.Patch(r, e, offset, value, size);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta: // non-numeric fields can't delta — copy
            for (const auto e : r.view<Selected>())
                if (patcher.Has(r, e)) patcher.Patch(r, e, offset, value, size);
            break;
    }
}

void ForEachSelectedWith(entt::registry &r, entt::id_type component_type, const std::function<void(entt::entity)> &fn) {
    const auto it = detail::PatchTable().find(component_type);
    if (it == detail::PatchTable().end()) return;
    const auto &has = it->second.Has;
    for (const auto e : r.view<Selected>())
        if (has(r, e)) fn(e);
}

void ApplyTagScoped(entt::registry &r, entt::entity viewport, Scope scope, entt::entity entity, entt::id_type tag_type, bool present) {
    switch (scope) {
        case Scope::Entity: ApplyTag(r, entity != null_entity ? entity : viewport, tag_type, present); break;
        case Scope::Active:
            if (const auto e = FindActiveEntity(r); e != null_entity) ApplyTag(r, e, tag_type, present);
            break;
        case Scope::Selected:
        case Scope::SelectedDelta:
            for (const auto e : r.view<Selected>()) ApplyTag(r, e, tag_type, present);
            break;
    }
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
