#pragma once

#include <entt/entity/entity.hpp>

// Component for entities that render a mesh via instancing.
// References an entity with Mesh+MeshBuffers+ModelsBuffer components.
struct MeshInstance {
    entt::entity MeshEntity{entt::null};
};

// Get the mesh entity for rendering/buffer purposes.
// Returns the MeshEntity if this is a MeshInstance, otherwise returns the entity itself.
entt::entity GetMeshEntity(const entt::registry &, entt::entity);
