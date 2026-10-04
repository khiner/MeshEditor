#pragma once
#include "state/Entity.h"
#include <span>
#include <vector>

// The colliders that fit or cook from each mesh: those naming it, and those on its instances.
struct MeshColliders {
    std::unordered_map<state::Entity, std::vector<state::Entity>> ByMesh;

    std::span<const state::Entity> Of(state::Entity mesh_entity) const {
        const auto it = ByMesh.find(mesh_entity);
        return it != ByMesh.end() ? std::span<const state::Entity>{it->second} : std::span<const state::Entity>{};
    }
};

// Rebuilds the mesh collider index when a collider was created, destroyed or moved to another mesh.
void UpdateMeshColliders(state::Scene &);
// Rederives each entity's collider shape, fitting auto-fit dimensions to its mesh's vertex bounds.
void RederiveColliders(state::Scene &, std::span<const state::Entity>);
