#include "armature/ArmatureComponents.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include <entt/entity/registry.hpp>

void Select(entt::registry &r, entt::entity e) {
    r.clear<Selected>();
    if (e != entt::null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void SelectBone(entt::registry &r, entt::entity e) {
    r.clear<BoneSelection>();
    if (e != entt::null) {
        r.clear<BoneActive>();
        r.emplace<BoneActive>(e);
        r.emplace<BoneSelection>(e);
    }
}
