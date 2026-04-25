// Bridge between gltf::Scene (CPU intermediate) and the EnTT registry + canonical UMA buffers.
//
// `PopulateGltfScene` populates registry components and writes to canonical UMA buffers
// (Materials, MeshStore arenas, TextureStore). Texture uploads and IBL prefilter currently run
// synchronously here; deferring them to `ProcessComponentEvents` via `Pending*` markers is a
// follow-up (Phase 3b).
//
// `BuildGltfScene` is the inverse: reads back into a fresh gltf::Scene suitable for SaveScene.

#pragma once

#include "GltfScene.h"
#include "SceneVulkanResources.h"
#include "entt_fwd.h"

#include <expected>
#include <filesystem>
#include <string>

#include <vulkan/vulkan.hpp>

struct DescriptorSlots;
struct EnvironmentStore;
struct MeshStore;
struct SceneBuffers;
struct TextureStore;

namespace gltf {

struct PopulateContext {
    entt::registry &R;
    entt::entity SceneEntity;
    SceneVulkanResources Vk;
    vk::CommandPool CommandPool;
    vk::Fence OneShotFence;
    DescriptorSlots &Slots;
    SceneBuffers &Buffers;
    MeshStore &Meshes;
    TextureStore &Textures;
    EnvironmentStore &Environments;
};

struct PopulateResult {
    entt::entity FirstMesh{null_entity};
    entt::entity Active{null_entity};
    entt::entity FirstCameraObject{null_entity};
    bool ImportedAnimation{false};
};

// `source` is mutated (mesh data is moved out into MeshStore arenas).
std::expected<PopulateResult, std::string>
PopulateGltfScene(Scene &source, const std::filesystem::path &source_path, PopulateContext ctx);

Scene BuildGltfScene(const entt::registry &R, entt::entity scene_entity);

} // namespace gltf
