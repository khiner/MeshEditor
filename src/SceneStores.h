#pragma once

#include "SceneVulkanResources.h"
#include "entt_fwd.h"

#include <memory>

#include <vulkan/vulkan.hpp>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct SceneBuffers;
struct TextureStore;

// Owns GPU-bound stores that the gltf populate/build path operates on. Construction uploads
// the default white texture (used as a material fallback) via the provided command pool +
// fence, but does not retain those handles.
struct SceneStores {
    SceneStores(SceneVulkanResources, vk::CommandPool, vk::Fence);
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
