#pragma once
#include "state/Entity.h"

#include <unordered_map>

namespace state {
struct Scene;
} // namespace state

// Derived: the entity holding each store record's mesh handle or evaluated mesh.
struct MeshEntities {
    std::unordered_map<uint32_t, state::Entity> ByStore;
};

void RegisterMeshStoreHandlers(state::Scene &);
// The mesh entity whose handle or evaluated mesh names the store record, or null.
state::Entity MeshEntityOf(const state::Scene &, uint32_t store_id);
