#pragma once
#include "state/Entity.h"

#include <vector>

// Derived: the entity drawing each store record, by store ID, or null.
struct MeshEntities {
    std::vector<state::Entity> ByStore;
};
inline void Reset(MeshEntities &entities, state::Registry &) { entities.ByStore.clear(); }

void RegisterMeshStoreHandlers(state::Registry &);
// The entity drawing the store record: the mesh entity evaluating to it, or the owned mesh entity holding it, or null.
state::Entity MeshEntityOf(const state::Registry &, uint32_t store_id);
