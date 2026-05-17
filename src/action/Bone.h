#pragma once

#include "numeric/mat4.h"
#include "numeric/quat.h"
#include "numeric/vec3.h"

#include <entt/entity/fwd.hpp>

#include <cstdint>
#include <memory>
#include <variant>

namespace action::bone {
struct Add {};
struct Extrude {};
struct DuplicateSelected {};
struct DeleteSelected {};
struct ClearSelectedTransforms {
    bool Position{false}, Rotation{false}, Scale{false};
};
// BoneDisplayScale is written without firing the reactive.
struct SetEditHeadTailRoll {
    entt::entity Entity;
    vec3 LocalP;
    quat LocalR;
    float DisplayScale;
};
struct SetConstraintTarget {
    entt::entity Entity;
    uint32_t Index;
    entt::entity Target;
};
struct SetConstraintInfluence {
    entt::entity Entity;
    uint32_t Index;
    float Influence;
};
struct SetConstraintChildOfInverse {
    entt::entity Entity;
    uint32_t Index;
    std::unique_ptr<mat4> Inverse;
};
struct DeleteConstraint {
    entt::entity Entity;
    uint32_t Index;
};
enum class BoneConstraintKind : uint8_t { CopyTransforms,
                                          ChildOf };
struct AddConstraint {
    entt::entity Entity;
    BoneConstraintKind Kind;
};

using Actions = std::variant<
    Add, Extrude, DuplicateSelected, DeleteSelected, ClearSelectedTransforms,
    SetEditHeadTailRoll,
    SetConstraintTarget, SetConstraintInfluence, SetConstraintChildOfInverse,
    DeleteConstraint, AddConstraint>;
} // namespace action::bone
