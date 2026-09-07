#include "mesh/MeshStores.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include <entt/entity/registry.hpp>
void RegisterMeshStoreHandlers(entt::registry &r) {
    r.on_destroy<MeshHandle>().connect<[](entt::registry &r, entt::entity e) {
        r.ctx().get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
}
void InitMeshStore(entt::registry &r, mtl::BufferContext &ctx) { r.ctx().emplace<MeshStore>(ctx); }
void ClearMeshStoreHandles(entt::registry &r) { r.clear<MeshHandle>(); }
void DeinitMeshStore(entt::registry &r) { r.ctx().erase<MeshStore>(); }
