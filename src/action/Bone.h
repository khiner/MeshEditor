#pragma once

#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

namespace action::bone {
struct Add {};
struct Extrude {};
struct DuplicateSelected {};
struct DeleteSelected {};
struct ClearSelectedTransforms {
    bool Position{false}, Rotation{false}, Scale{false};
};
// All of these target the active bone (FindActiveBone(R)).
struct SetEditHeadTailRoll {
    vec3 LocalP;
    quat LocalR;
    float DisplayScale;
};
struct SetConstraintTarget {
    uint32_t Index;
    entt::entity Target;
};
struct SetConstraintInfluence {
    uint32_t Index;
    float Influence;
};
struct SetConstraintChildOfInverse {
    uint32_t Index;
    std::unique_ptr<mat4> Inverse;
};
struct DeleteConstraint {
    uint32_t Index;
};
enum class BoneConstraintKind : uint8_t { CopyTransforms,
                                          ChildOf };
struct AddConstraint {
    BoneConstraintKind Kind;
};

using Actions = std::variant<
    Add, Extrude, DuplicateSelected, DeleteSelected, ClearSelectedTransforms,
    SetEditHeadTailRoll,
    SetConstraintTarget, SetConstraintInfluence, SetConstraintChildOfInverse,
    DeleteConstraint, AddConstraint>;
using Action = Actions;

void Apply(entt::registry &, entt::entity viewport, const Action &);
} // namespace action::bone
