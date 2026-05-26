#include "selection/SelectionOps.h"

#include "scene/Entity.h" // Selected, Active

#include <entt/entity/registry.hpp>

void Select(entt::registry &r, entt::entity e) {
    r.clear<Selected>();
    if (e != entt::null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void ToggleSelected(entt::registry &r, entt::entity e) {
    if (e == entt::null) return;
    if (r.all_of<Selected>(e)) r.remove<Selected>(e);
    else r.emplace_or_replace<Selected>(e);
}
