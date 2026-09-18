#pragma once

#include "numeric/quat.h"
#include "numeric/vec3.h"

#include "state/Entity.h"

namespace action::bone {
using numeric::quat, numeric::vec3;

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
    state::Entity Target;
};
struct SetConstraintInfluence {
    uint32_t Index;
    float Influence;
};
// Bake inverse(target_world) * bone_world so the current relative pose becomes the constraint's rest offset.
struct BakeConstraintChildOfInverse {
    uint32_t Index;
};
// Reset the Child-Of inverse to identity (no offset).
struct ClearConstraintChildOfInverse {
    uint32_t Index;
};
struct DeleteConstraint {
    uint32_t Index;
};
enum class BoneConstraintKind : uint8_t { CopyTransforms,
                                          ChildOf };
struct AddConstraint {
    BoneConstraintKind Kind;
};

using Action = std::variant<
    Add, Extrude, DuplicateSelected, DeleteSelected, ClearSelectedTransforms,
    SetEditHeadTailRoll,
    SetConstraintTarget, SetConstraintInfluence,
    BakeConstraintChildOfInverse, ClearConstraintChildOfInverse,
    DeleteConstraint, AddConstraint>;

void Apply(state::Scene &, state::Entity viewport, const Action &);
} // namespace action::bone
