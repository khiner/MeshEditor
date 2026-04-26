#pragma once

#include "SceneVulkanResources.h"
#include "entt_fwd.h"

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

// Wires `registry` to `stores`: creates the scene singleton entity, emplaces required
// components (NameRegistry, ObjectIdCounter, MaterialStore, AnimationTimeline), registers
// the registry-only physics companion-component handlers (PhysicsMotion → PhysicsVelocity,
// ColliderShape → ColliderMaterial), and appends the default fallback material. Returns
// the scene singleton entity.
entt::entity WireSceneRegistry(entt::registry &, SceneStores &);
