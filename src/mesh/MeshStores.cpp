#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/PendingSync.h"
#include "state/Scene.h"

namespace {
void MapMeshEntity(state::Scene &r, state::Entity e) { r.Context.get<MeshEntities>().ByStore[r.get<const MeshHandle>(e).StoreId] = e; }
} // namespace

void RegisterMeshStoreHandlers(state::Scene &r) {
    r.Context.emplace<MeshEntities>();
    r.on_construct<MeshHandle, &MapMeshEntity>();
    r.on_update<MeshHandle, &MapMeshEntity>();
    r.on_destroy<MeshHandle, [](state::Scene &r, state::Entity e) {
        auto &entities = r.Context.get<MeshEntities>().ByStore;
        if (const auto it = entities.find(r.get<const MeshHandle>(e).StoreId); it != entities.end() && it->second == e) entities.erase(it);
    }>();
    r.on_destroy<MeshHandle, [](state::Scene &r, state::Entity e) {
        if (r.Restoring) return;
        r.Context.emplace<PendingObjectRemovals>().StoreIds.push_back(r.get<const MeshHandle>(e).StoreId);
    }>();
}

state::Entity MeshEntityOf(const state::Scene &r, uint32_t store_id) {
    const auto &entities = r.Context.get<const MeshEntities>().ByStore;
    const auto it = entities.find(store_id);
    if (it == entities.end() || !r.valid(it->second)) return state::Null;
    const auto *handle = r.try_get<const MeshHandle>(it->second);
    return handle && handle->StoreId == store_id ? it->second : state::Null;
}
