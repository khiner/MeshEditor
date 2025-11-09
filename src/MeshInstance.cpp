#include "MeshInstance.h"

#include <entt/entity/registry.hpp>

entt::entity GetMeshEntity(const entt::registry &r, entt::entity entity) {
    if (const auto *mesh_instance = r.try_get<MeshInstance>(entity)) {
        return mesh_instance->MeshEntity;
    }
    return entity;
}
