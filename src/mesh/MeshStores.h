#pragma once
#include "state/Entity.h"

#include <vector>

namespace state {
struct Scene;
} // namespace state

// Derived: the entity holding each store record's mesh handle or evaluated mesh, by store ID, or null.
struct MeshEntities {
    std::vector<state::Entity> ByStore;
};
inline void Reset(MeshEntities &entities, state::Scene &) { entities.ByStore.clear(); }

void RegisterMeshStoreHandlers(state::Scene &);
// The mesh entity whose handle or evaluated mesh names the store record, or null.
state::Entity MeshEntityOf(const state::Scene &, uint32_t store_id);
