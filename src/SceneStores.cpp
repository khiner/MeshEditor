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
#include <memory>
#include <vector>

#include <entt/entity/registry.hpp>

void InitSceneStoreCtx(entt::registry &r, SceneVulkanResources vk) {
    r.ctx().emplace<SceneVulkanResources>(vk);
    auto &slots = r.ctx().emplace<DescriptorSlots>(
        vk.Device,
        vk.PhysicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>().get<vk::PhysicalDeviceDescriptorIndexingProperties>()
    );
    auto &textures = *r.ctx().emplace<std::unique_ptr<TextureStore>>(std::make_unique<TextureStore>());
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.ctx().emplace<std::unique_ptr<EnvironmentStore>>(std::make_unique<EnvironmentStore>());
}

entt::entity WireSceneRegistry(entt::registry &r, SceneVulkanResources vk) {
    r.on_construct<PhysicsMotion>().connect<&entt::registry::emplace<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&entt::registry::remove<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&entt::registry::emplace<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&entt::registry::remove<ColliderMaterial>>();

    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &textures = *r.ctx().get<std::unique_ptr<TextureStore>>();

    const auto scene_entity = r.create();
    auto &buffers = r.emplace<SceneBuffers>(scene_entity, vk.PhysicalDevice, vk.Device, vk.Instance, slots);
    r.ctx().emplace<MeshStore>(buffers.Ctx);

    r.emplace<NameRegistry>(scene_entity);
    r.emplace<ObjectIdCounter>(scene_entity);
    r.emplace<MaterialStore>(scene_entity);
    r.emplace<TimelineRange>(scene_entity);
    r.emplace<TimelinePlayback>(scene_entity);

    buffers.Materials.Append({
        .BaseColorFactor = vec4{1.f},
        .MetallicFactor = 0.f,
        .RoughnessFactor = 1.f,
        .AlphaMode = MaterialAlphaMode::Opaque,
        .AlphaCutoff = 0.5f,
        .DoubleSided = 0u,
        .BaseColorTexture = {.Slot = textures.WhiteTextureSlot},
    });
    r.patch<MaterialStore>(scene_entity, [](auto &m) { m.Names.emplace_back("Default"); });

    constexpr std::array<std::byte, 4> WhitePixels{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    auto &pending = r.get_or_emplace<PendingTextureUploads>(scene_entity);
    pending.Items.emplace_back(PendingTextureUpload{
        .SamplerSlot = textures.WhiteTextureSlot,
        .Source = PendingTextureUpload::RawPixels{.Pixels = std::vector<std::byte>(WhitePixels.begin(), WhitePixels.end()), .Width = 1, .Height = 1},
        .ColorSpace = TextureColorSpace::Srgb,
        .WrapS = vk::SamplerAddressMode::eRepeat,
        .WrapT = vk::SamplerAddressMode::eRepeat,
        .Sampler = SamplerConfig{},
        .Name = "DefaultWhite",
    });

    return scene_entity;
}

void TearDownSceneStoreCtx(entt::registry &r) {
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &textures = *r.ctx().get<std::unique_ptr<TextureStore>>();
    auto &environments = *r.ctx().get<std::unique_ptr<EnvironmentStore>>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseSamplerSlots(slots, CollectSamplerSlots(textures.Textures));

    // Tear down GPU-resource owners before SceneBuffers, since they retire allocations into
    // SceneBuffers.Ctx.Retired (cleared on ~BufferContext, which destroys the VMA allocator).
    r.ctx().erase<std::unique_ptr<EnvironmentStore>>();
    r.ctx().erase<std::unique_ptr<TextureStore>>();
    r.ctx().erase<MeshStore>();
    // Destroy entities owning a SceneBuffers component so it drops (and its ~BufferContext destroys
    // the VMA allocator) before DescriptorSlots is erased.
    std::vector<entt::entity> scene_entities;
    for (auto e : r.view<SceneBuffers>()) scene_entities.push_back(e);
    for (auto e : scene_entities) r.destroy(e);
    r.ctx().erase<DescriptorSlots>();
    r.ctx().erase<SceneVulkanResources>();
}
