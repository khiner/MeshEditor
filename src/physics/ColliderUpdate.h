#pragma once
#include "state/Entity.h"
#include <span>
#include <vector>

// The colliders that fit or cook from a mesh: those naming it, and those on its instances.
struct MeshColliderList {
    std::vector<state::Entity> Colliders;
};
// The mesh whose collider list contains this collider.
// The destroy handler removes the collider from that list.
struct ColliderListedUnder {
    state::Entity MeshEntity;
};

void RegisterMeshColliderHandlers(state::Scene &);
// The colliders listed under the mesh.
std::span<const state::Entity> MeshCollidersOf(const state::Scene &, state::Entity mesh_entity);
// Moves each created or changed collider to its current mesh's list, and takes each collider whose shape left out of its list.
void UpdateMeshColliders(state::Scene &);
// Rederives each entity's collider shape, fitting auto-fit dimensions to its mesh's vertex bounds.
void RederiveColliders(state::Scene &, std::span<const state::Entity>);
