#include "armature/ArmatureComponents.h"
#include "scene/Entity.h"
#include "selection/BoneSelection.h"
#include "selection/SelectionComponents.h"
#include "selection/SelectionOps.h"
#include "state/Scene.h"

void Select(state::Scene &r, state::Entity e) {
    r.clear<Selected>();
    if (e != state::Null) {
        r.clear<Active>();
        r.emplace<Active>(e);
        r.emplace<Selected>(e);
    }
}

void SelectBone(state::Scene &r, state::Entity e) {
    r.clear<BoneSelection>();
    if (e != state::Null) {
        r.clear<BoneActive>();
        r.emplace<BoneActive>(e);
        r.emplace<BoneSelection>(e);
    }
}
