#pragma once

#include "numeric/mat4.h"
#include "state/Entity.h"

#include <variant>
#include <vector>

// Pose constraint stack on bone entities.
struct CopyTransformsData {};
struct ChildOfData {
    mat4 InverseMatrix{I4}; // Stored "parent-inverse" like Blender's Child Of: inverse(target_world) * owner_world at bind time.
};

struct BoneConstraint {
    state::Entity TargetEntity{null_entity};
    float Influence{1.f};
    std::variant<CopyTransformsData, ChildOfData> Data{CopyTransformsData{}};
};
struct BoneConstraints {
    std::vector<BoneConstraint> Stack;
};

// User-authored pose constraint stack.
