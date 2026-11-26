#pragma once
#include "state/Entity.h"
#include <optional>
#include <span>
#include <vector>

struct AABB;
struct ColliderShape;

// The colliders that fit or cook from a mesh: those naming it, and those on its instances.
struct MeshColliderList {
    std::vector<state::Entity> Colliders;
};
// The mesh whose collider list contains this collider.
// The destroy handler removes the collider from that list.
struct ColliderListedUnder {
    state::Entity MeshEntity;
};

void RegisterMeshColliderHandlers(state::Registry &);
// The colliders listed under the mesh.
std::span<const state::Entity> MeshCollidersOf(const state::Registry &, state::Entity mesh_entity);
// Moves each created or changed collider to its current mesh's list, and takes each collider whose shape left out of its list.
void UpdateMeshColliders(state::Registry &);
// The collider's shape derived from its mesh, fitting auto-fit dimensions to the vertex bounds of a mesh with vertices.
std::optional<ColliderShape> DeriveColliderShape(const state::Registry &, state::Entity collider, std::optional<AABB> vertex_bounds);
// Rederives each entity's collider shape, fitting auto-fit dimensions to its mesh's vertex bounds.
void RederiveColliders(state::Registry &, std::span<const state::Entity>);
