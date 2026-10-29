#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/PendingSync.h"
#include "state/Registry.h"

namespace {
// Whether the entity draws the record as its evaluated or held record.
bool Draws(const state::Registry &r, state::Entity e, uint32_t id) {
    if (const auto *evaluated = r.try_get<const EvaluatedMesh>(e); evaluated && evaluated->StoreId == id) return true;
    if (const auto *handle = r.try_get<const MeshHandle>(e); handle && handle->StoreId == id) return true;
    const auto *vertices = r.try_get<const VertexStoreId>(e);
    return vertices && vertices->StoreId == id;
}
// Each record an entity draws maps to the entity.
template<typename C> void MapMeshEntity(state::Registry &r, state::Entity e) {
    auto &entities = r.Resources.get<MeshEntities>().ByStore;
    const auto id = r.get<const C>(e).StoreId;
    if (id >= entities.size()) entities.resize(id + 1, state::Null);
    entities[id] = e;
}
template<typename C> void UnmapMeshEntity(state::Registry &r, state::Entity e) {
    auto &entities = r.Resources.get<MeshEntities>().ByStore;
    if (const auto id = r.get<const C>(e).StoreId; id < entities.size() && entities[id] == e) entities[id] = state::Null;
}
// A record its owner drops outside a restore releases at the next settle.
template<typename C> void QueueRelease(state::Registry &r, state::Entity e) {
    if (r.Restoring) return;
    r.Resources.emplace<PendingObjectRemovals>().StoreIds.push_back(r.get<const C>(e).StoreId);
}
template<typename C> void TrackRecord(state::Registry &r) {
    r.on_construct<C, &MapMeshEntity<C>>();
    r.on_update<C, &MapMeshEntity<C>>();
    r.on_destroy<C, &UnmapMeshEntity<C>>();
    r.on_destroy<C, &QueueRelease<C>>();
}
} // namespace

void RegisterMeshStoreHandlers(state::Registry &r) {
    r.Resources.emplace<MeshEntities>();
    TrackRecord<MeshHandle>(r);
    TrackRecord<EvaluatedMesh>(r);
    TrackRecord<VertexStoreId>(r);
}

state::Entity MeshEntityOf(const state::Registry &r, uint32_t store_id) {
    const auto &entities = r.Resources.get<const MeshEntities>().ByStore;
    const auto e = store_id < entities.size() ? entities[store_id] : state::Null;
    return r.valid(e) && Draws(r, e, store_id) ? e : state::Null;
}

const MeshStore::Record *TryRecordOf(const state::Registry &r, state::Entity e) {
    const auto id = DrawnStoreId(r, e);
    return id ? r.Resources.get<const MeshStore>().TryGet(*id) : nullptr;
}
const MeshStore::Record &RecordOf(const state::Registry &r, state::Entity e) { return r.Resources.get<const MeshStore>().Get(*DrawnStoreId(r, e)); }
MeshStore::Record &EditRecordOf(state::Registry &r, state::Entity e) { return r.Resources.get<MeshStore>().WriteRecord(*DrawnStoreId(r, e)); }
