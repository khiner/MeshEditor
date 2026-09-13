#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "project/Registry.h"
#include <entt/entity/registry.hpp>
void RegisterMeshStoreHandlers(entt::registry &r) {
    r.on_destroy<MeshHandle>().connect<[](entt::registry &r, entt::entity e) {
        if (project::Restoring(r)) return;
        r.ctx().get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
}
void InitMeshStore(entt::registry &r, mtl::BufferContext &ctx) { r.ctx().emplace<MeshStore>(ctx); }
void ClearMeshStoreHandles(entt::registry &r) { project::Clear<MeshHandle>(r); }
void DeinitMeshStore(entt::registry &r) { r.ctx().erase<MeshStore>(); }
