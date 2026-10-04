#pragma once

#include "numeric/mat4.h"
#include "state/Entity.h"

#include <variant>
#include <vector>

// Pose constraint stack on bone entities.
using numeric::I4, numeric::mat4;

struct CopyTransformsData {};
struct ChildOfData {
    mat4 InverseMatrix{I4}; // Stored "parent-inverse" like Blender's Child Of: inverse(target_world) * owner_world at bind time.
};

struct BoneConstraint {
    state::Entity TargetEntity{state::Null};
    float Influence{1.f};
    std::variant<CopyTransformsData, ChildOfData> Data{CopyTransformsData{}};
};
struct BoneConstraints {
    std::vector<BoneConstraint> Stack;
};

// The armature objects whose bone constraints read each entity's world transform: every constraint target, and each constrained armature object itself.
// Derived from the constraint stacks.
struct ConstraintTargets {
    std::unordered_map<state::Entity, std::vector<state::Entity>> Armatures;
};

// User-authored pose constraint stack.
