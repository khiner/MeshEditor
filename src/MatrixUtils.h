#pragma once

#include "numeric/mat4.h"

#include <entt/entity/fwd.hpp>

struct Transform;

// Stores the spatial offset from parent at parenting time
// Formula: W_child = W_parent × ParentInverse × L_child
struct ParentInverse {
    mat4 Value{I4};
};

// Get decomposed transform from entity components
Transform GetTransform(const entt::registry &, entt::entity);

// Compute local-space matrix from decomposed transform
mat4 ToLocalMatrix(const Transform &);

// Compute world-space matrix for entity (accounting for parent hierarchy)
mat4 ComputeWorldMatrix(const entt::registry &, entt::entity);
