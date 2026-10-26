#pragma once

// Registry-mutating scene-graph operations: reparenting.

// Unlinks each child from its parent, walking each affected sibling list once.
void ClearParents(state::Registry &, std::span<const state::Entity> children);

// Links a parentless child at the head of the parent's children.
// The child keeps its local Transform, so its world becomes the parent's world times its Transform.
void SetParent(state::Registry &, state::Entity child, state::Entity parent);

// Links the children under the parent, preserving each child's world pose by decomposing inverse(parent_world)*old_world into Transform.
// An owned node's world pose is its published world transform, as in ComposeTransform.
// The parent itself and its ancestors stay where they are.
// Returns whether an ancestor of the parent was among the children.
// Nonuniform parent scaling makes this decomposition lossy, as in BKE_object_apply_parent_inverse.
bool SetParentKeepWorld(state::Registry &, std::span<const state::Entity> children, state::Entity parent, std::span<const state::Entity> owned);
