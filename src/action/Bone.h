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
// A null BoneEntity targets the active bone.
struct SetEditHeadTailRoll {
    vec3 LocalP;
    quat LocalR;
    float DisplayScale;
    state::Entity BoneEntity{state::Null};
};
struct SetConstraintTarget {
    uint32_t Index;
    state::Entity Target;
    state::Entity BoneEntity{state::Null};
};
struct SetConstraintInfluence {
    uint32_t Index;
    float Influence;
    state::Entity BoneEntity{state::Null};
};
// Bake inverse(target_world) * bone_world so the current relative pose becomes the constraint's rest offset.
struct BakeConstraintChildOfInverse {
    uint32_t Index;
    state::Entity BoneEntity{state::Null};
};
struct ClearConstraintChildOfInverse {
    uint32_t Index;
    state::Entity BoneEntity{state::Null};
};
struct DeleteConstraint {
    uint32_t Index;
    state::Entity BoneEntity{state::Null};
};
enum class BoneConstraintKind : uint8_t { CopyTransforms,
                                          ChildOf };
struct AddConstraint {
    BoneConstraintKind Kind;
    state::Entity BoneEntity{state::Null};
};

using Action = std::variant<
    Add, Extrude, DuplicateSelected, DeleteSelected, ClearSelectedTransforms,
    SetEditHeadTailRoll,
    SetConstraintTarget, SetConstraintInfluence,
    BakeConstraintChildOfInverse, ClearConstraintChildOfInverse,
    DeleteConstraint, AddConstraint>;

void Apply(state::Registry &, const Action &);
} // namespace action::bone
