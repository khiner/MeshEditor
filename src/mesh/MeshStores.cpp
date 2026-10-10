#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/PendingSync.h"
#include "state/Scene.h"

namespace {
// Whether the entity's other record component, its source handle or its evaluated mesh, names the record C names.
template<typename C> bool NamedByOther(const state::Scene &r, state::Entity e) {
    using Other = std::conditional_t<std::same_as<C, MeshHandle>, EvaluatedMesh, MeshHandle>;
    const auto *other = r.try_get<const Other>(e);
    return other && other->StoreId == r.get<const C>(e).StoreId;
}
// Each record a mesh entity owns, its source and its evaluated output, maps to the entity.
template<typename C> void MapMeshEntity(state::Scene &r, state::Entity e) { r.Context.get<MeshEntities>().ByStore[r.get<const C>(e).StoreId] = e; }
template<typename C> void UnmapMeshEntity(state::Scene &r, state::Entity e) {
    auto &entities = r.Context.get<MeshEntities>().ByStore;
    if (const auto it = entities.find(r.get<const C>(e).StoreId); it != entities.end() && it->second == e && !NamedByOther<C>(r, e)) entities.erase(it);
}
// A record its owner drops outside a restore releases at the next settle, unless the owner still names it.
template<typename C> void QueueRelease(state::Scene &r, state::Entity e) {
    if (r.Restoring || NamedByOther<C>(r, e)) return;
    r.Context.emplace<PendingObjectRemovals>().StoreIds.push_back(r.get<const C>(e).StoreId);
}
template<typename C> void TrackRecord(state::Scene &r) {
    r.on_construct<C, &MapMeshEntity<C>>();
    r.on_update<C, &MapMeshEntity<C>>();
    r.on_destroy<C, &UnmapMeshEntity<C>>();
    r.on_destroy<C, &QueueRelease<C>>();
}
} // namespace

void RegisterMeshStoreHandlers(state::Scene &r) {
    r.Context.emplace<MeshEntities>();
    TrackRecord<MeshHandle>(r);
    TrackRecord<EvaluatedMesh>(r);
}

state::Entity MeshEntityOf(const state::Scene &r, uint32_t store_id) {
    const auto &entities = r.Context.get<const MeshEntities>().ByStore;
    const auto it = entities.find(store_id);
    if (it == entities.end() || !r.valid(it->second)) return state::Null;
    const auto *handle = r.try_get<const MeshHandle>(it->second);
    const auto *evaluated = r.try_get<const EvaluatedMesh>(it->second);
    return (handle && handle->StoreId == store_id) || (evaluated && evaluated->StoreId == store_id) ? it->second : state::Null;
}

const MeshStore::Record *TryRecordOf(const state::Scene &r, state::Entity e) {
    const auto id = DrawnStoreId(r, e);
    return id ? r.Context.get<const MeshStore>().TryGet(*id) : nullptr;
}
const MeshStore::Record &RecordOf(const state::Scene &r, state::Entity e) { return r.Context.get<const MeshStore>().Get(*DrawnStoreId(r, e)); }
MeshStore::Record &EditRecordOf(state::Scene &r, state::Entity e) { return r.Context.get<MeshStore>().WriteRecord(*DrawnStoreId(r, e)); }
