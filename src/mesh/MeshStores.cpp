#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "state/Scene.h"
void RegisterMeshStoreHandlers(state::Scene &r) {
    r.on_destroy<MeshHandle>().connect<[](state::Scene &r, state::Entity e) {
        if (r.Restoring) return;
        r.ctx().get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
}
void InitMeshStore(state::Scene &r, mtl::BufferContext &ctx) { r.ctx().emplace<MeshStore>(ctx); }
void ClearMeshStoreHandles(state::Scene &r) { r.clear<MeshHandle>(); }
void DeinitMeshStore(state::Scene &r) { r.ctx().erase<MeshStore>(); }
