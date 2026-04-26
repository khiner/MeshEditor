#include "SceneStores.h"

#include "AnimationTimeline.h"
#include "Bindless.h"
#include "SceneOps.h"
#include "SceneTextures.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsTypes.h"
#include "scene_impl/SceneBuffers.h"

#include <array>
#include <cstddef>

#include <entt/entity/registry.hpp>

SceneStores::SceneStores(SceneVulkanResources vk)
    : Slots{std::make_unique<DescriptorSlots>(
          vk.Device,
          vk.PhysicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>().get<vk::PhysicalDeviceDescriptorIndexingProperties>()
      )},
      Buffers{std::make_unique<SceneBuffers>(vk.PhysicalDevice, vk.Device, vk.Instance, *Slots)},
      Meshes{std::make_unique<MeshStore>(Buffers->Ctx)},
      Textures{std::make_unique<TextureStore>()},
      Environments{std::make_unique<EnvironmentStore>()} {
    Textures->WhiteTextureSlot = AllocateSamplerSlot(*Slots);
}

SceneStores::~SceneStores() {
    if (Environments && Slots) ReleaseEnvironmentSamplerSlots(*Slots, *Environments);
    if (Textures && Slots) ReleaseSamplerSlots(*Slots, CollectSamplerSlots(Textures->Textures));
}

entt::entity WireSceneRegistry(entt::registry &r, SceneStores &stores) {
    r.on_construct<PhysicsMotion>().connect<&entt::registry::emplace<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&entt::registry::remove<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&entt::registry::emplace<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&entt::registry::remove<ColliderMaterial>>();

    const auto scene_entity = r.create();
    r.emplace<NameRegistry>(scene_entity);
    r.emplace<ObjectIdCounter>(scene_entity);
    r.emplace<MaterialStore>(scene_entity);
    r.emplace<AnimationTimeline>(scene_entity);

    stores.Buffers->Materials.Append({
        .BaseColorFactor = vec4{1.f},
        .MetallicFactor = 0.f,
        .RoughnessFactor = 1.f,
        .AlphaMode = MaterialAlphaMode::Opaque,
        .AlphaCutoff = 0.5f,
        .DoubleSided = 0u,
        .BaseColorTexture = {.Slot = stores.Textures->WhiteTextureSlot},
    });
    r.patch<MaterialStore>(scene_entity, [](auto &m) { m.Names.emplace_back("Default"); });

    constexpr std::array<std::byte, 4> WhitePixels{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    auto &pending = r.get_or_emplace<PendingTextureUploads>(scene_entity);
    pending.Items.emplace_back(PendingTextureUpload{
        .SamplerSlot = stores.Textures->WhiteTextureSlot,
        .Source = PendingTextureUpload::RawPixels{.Pixels = std::vector<std::byte>(WhitePixels.begin(), WhitePixels.end()), .Width = 1, .Height = 1},
        .ColorSpace = TextureColorSpace::Srgb,
        .WrapS = vk::SamplerAddressMode::eRepeat,
        .WrapT = vk::SamplerAddressMode::eRepeat,
        .Sampler = SamplerConfig{},
        .Name = "DefaultWhite",
    });

    return scene_entity;
}
