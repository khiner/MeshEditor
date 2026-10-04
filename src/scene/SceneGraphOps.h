#pragma once

#include "state/Entity.h"

// Registry-mutating scene-graph operations: reparenting.

// Unlinks each child from its parent, walking each affected sibling list once.
void ClearParents(state::Scene &, std::span<const state::Entity> children);

// Links a parentless child at the head of the parent's children.
// The child keeps its local Transform, so its world becomes the parent's world times its Transform.
void SetParent(state::Scene &, state::Entity child, state::Entity parent);

// Links the children under the parent, preserving each child's world pose by decomposing inverse(parent_world)*old_world into Transform.
// The parent itself and its ancestors stay where they are.
// Returns whether an ancestor of the parent was among the children.
// Nonuniform parent scaling makes this decomposition lossy, as in BKE_object_apply_parent_inverse.
bool SetParentKeepWorld(state::Scene &, std::span<const state::Entity> children, state::Entity parent);
