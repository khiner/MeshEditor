#include "action/Core.h"
#include "Variant.h"
#include "action/Dispatch.h"
#include "scene/Entity.h"

#include <entt/entity/registry.hpp>

namespace action {
void Apply(entt::registry &r, entt::entity viewport, const Core &action) {
    std::visit(
        overloaded{
            [&]<typename Field>(const Update<Field> &a) { ApplyUpdate(r, viewport, a); },
            [&]<typename Field>(const UpdateActive<Field> &a) { ApplyUpdate(r, FindActiveEntity(r), a.ComponentType, a.Offset, a.Value); },
            [&](const SetTag &a) { ApplyTag(r, a.Entity, a.TagType, a.Present); },
            [&](const SetActiveTag &a) { ApplyTag(r, FindActiveEntity(r), a.TagType, a.Present); },
            [&](const DestroyEntity &a) { r.destroy(a.Entity); },
        },
        action
    );
}
} // namespace action
