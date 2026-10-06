#pragma once
#include "state/Entity.h"

#include <unordered_map>

namespace state {
struct Scene;
}

// Derived: the entity holding each store record's mesh handle.
struct MeshEntities {
    std::unordered_map<uint32_t, state::Entity> ByStore;
};

void RegisterMeshStoreHandlers(state::Scene &);
// The mesh entity whose handle names the store record, or null.
state::Entity MeshEntityOf(const state::Scene &, uint32_t store_id);
