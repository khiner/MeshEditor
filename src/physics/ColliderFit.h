#pragma once

#include "PhysicsTypes.h"

struct Mesh;

entt::entity FindMeshEntity(const entt::registry &, entt::entity);

// Re-derives `e`'s ColliderShape from its Mesh + PrimitiveShape + ColliderPolicy.
// No-op if any input is missing. Always patches when present.
void RederiveCollider(entt::registry &, entt::entity);
