#pragma once

#include "SceneVulkanResources.h"

#include <entt/entity/fwd.hpp>

#include <memory>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct SceneBuffers;
struct TextureStore;

struct SceneStores {
    SceneStores(SceneVulkanResources);
    ~SceneStores();
    SceneStores(const SceneStores &) = delete;
    SceneStores &operator=(const SceneStores &) = delete;

    std::unique_ptr<DescriptorSlots> Slots;
    std::unique_ptr<SceneBuffers> Buffers;
    std::unique_ptr<MeshStore> Meshes;
    std::unique_ptr<TextureStore> Textures;
    std::unique_ptr<EnvironmentStore> Environments;
};

// Initializes the registry with scene-singleton state and returns its entity.
entt::entity WireSceneRegistry(entt::registry &, SceneStores &);
