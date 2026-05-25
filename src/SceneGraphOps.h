#pragma once

#include "entt_fwd.h"

// Registry-mutating scene-graph operations: reparenting and world-transform recompute.

void ClearParent(entt::registry &, entt::entity child);

// Snap: child keeps its local Transform; new world = parent_world * Transform.
void SetParent(entt::registry &, entt::entity child, entt::entity parent);

// Keep-world: child preserves world pose by decomposing inv(parent_world) * old_world into
// Transform. Lossy under non-uniform parent scale (cf. BKE_object_apply_parent_inverse).
void SetParentKeepWorld(entt::registry &, entt::entity child, entt::entity parent);

// Recompute WT for `e` and its descendants from local Transforms and ancestor's WT.
void UpdateWorldTransformRecursive(entt::registry &, entt::entity e);
