#pragma once

#include "mesh/MeshComponents.h" // RenderInstance

#include <entt/entity/registry.hpp>

// Unmanaged reactive storage tracking entity destruction.
// Unlike managed storage (via r.storage<entt::reactive>()), this keeps destroyed entities until manually cleared, so deletions stay observable across a frame.
struct EntityDestroyTracker {
    entt::storage_for_t<entt::reactive> Storage;

    void Bind(entt::registry &r) {
        Storage.bind(r);
        Storage.on_destroy<RenderInstance>();
    }
};
