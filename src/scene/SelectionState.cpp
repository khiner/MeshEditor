#include "armature/ArmatureComponents.h"
#include "project/Registry.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include <entt/entity/registry.hpp>

void Select(entt::registry &r, entt::entity e) {
    project::Clear<Selected>(r);
    if (e != entt::null) {
        project::Clear<Active>(r);
        project::Emplace<Active>(r, e);
        project::Emplace<Selected>(r, e);
    }
}

void SelectBone(entt::registry &r, entt::entity e) {
    project::Clear<BoneSelection>(r);
    if (e != entt::null) {
        project::Clear<BoneActive>(r);
        project::Emplace<BoneActive>(r, e);
        project::Emplace<BoneSelection>(r, e);
    }
}
