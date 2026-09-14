#pragma once

#include "state/Entity.h"

// Registry-mutating scene-graph operations: reparenting and world-transform recompute.

void ClearParent(state::Scene &, state::Entity child);

// Snap: child keeps its local Transform; new world = parent_world * Transform.
void SetParent(state::Scene &, state::Entity child, state::Entity parent);

// Preserves child world pose by decomposing inverse(parent_world)*old_world into Transform.
// Nonuniform parent scaling makes this decomposition lossy; see BKE_object_apply_parent_inverse.
void SetParentKeepWorld(state::Scene &, state::Entity child, state::Entity parent);

// Recompute WT for `e` and its descendants from local Transforms and ancestor's WT.
void UpdateWorldTransformRecursive(state::Scene &, state::Entity e);
