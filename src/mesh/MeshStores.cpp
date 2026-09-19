#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "state/Scene.h"
void RegisterMeshStoreHandlers(state::Scene &r) {
    r.on_destroy<MeshHandle, [](state::Scene &r, state::Entity e) {
        if (r.Restoring) return;
        r.Context.get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
}
