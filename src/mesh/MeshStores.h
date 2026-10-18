#pragma once
#include "mesh/MeshSchema.h"
#include "state/Entity.h"

#include <vector>

// Derived: the entity holding each store record's mesh handle or evaluated mesh, by store ID, or null.
struct MeshEntities {
    std::vector<state::Entity> ByStore;
};
inline void Reset(MeshEntities &entities, state::Registry &) { entities.ByStore.clear(); }

void RegisterMeshStoreHandlers(state::Registry &);
// The mesh entity whose handle or evaluated mesh names the store record, or null.
state::Entity MeshEntityOf(const state::Registry &, uint32_t store_id);
