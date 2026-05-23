#include "Stores.h"

#include "AnimationTimeline.h"
#include "Bindless.h"
#include "Entity.h"
#include "GpuBuffers.h"
#include "ObjectComponents.h"
#include "Textures.h"
#include "ViewportComponents.h"
#include "mesh/MeshStore.h"
#include "physics/PhysicsTypes.h"

void InitStoreCtx(entt::registry &r, VulkanResources vk) {
    r.ctx().emplace<VulkanResources>(vk);
    auto &slots = r.ctx().emplace<DescriptorSlots>(
        vk.Device,
        vk.PhysicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>().get<vk::PhysicalDeviceDescriptorIndexingProperties>()
    );
    auto &textures = r.ctx().emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.ctx().emplace<EnvironmentStore>();
}

entt::entity WireRegistry(entt::registry &r) {
    r.on_construct<PhysicsMotion>().connect<&entt::registry::emplace<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&entt::registry::remove<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&entt::registry::emplace<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&entt::registry::remove<ColliderMaterial>>();

    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &textures = r.ctx().get<TextureStore>();

    const auto viewport = r.create();
    auto &buffers = r.ctx().emplace<GpuBuffers>(vk.PhysicalDevice, vk.Device, vk.Instance, slots);
    r.ctx().emplace<MeshStore>(buffers.Ctx);

    r.ctx().emplace<NameRegistry>();
    r.ctx().emplace<ObjectIdCounter>();
    auto &materials = r.ctx().emplace<MaterialStore>();
    r.emplace<TimelineRange>(viewport);
    r.emplace<TimelinePlayback>(viewport);

    buffers.Materials.Append({
        .BaseColorFactor = vec4{1.f},
        .MetallicFactor = 0.f,
        .RoughnessFactor = 1.f,
        .AlphaMode = MaterialAlphaMode::Opaque,
        .AlphaCutoff = 0.5f,
        .DoubleSided = 0u,
        .BaseColorTexture = {.Slot = textures.WhiteTextureSlot},
    });
    materials.Names.emplace_back("Default");

    constexpr std::array<std::byte, 4> WhitePixels{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    auto &pending = r.get_or_emplace<PendingTextureUploads>(viewport);
    pending.Items.emplace_back(PendingTextureUpload{
        .SamplerSlot = textures.WhiteTextureSlot,
        .Source = PendingTextureUpload::RawPixels{.Pixels = std::vector<std::byte>(WhitePixels.begin(), WhitePixels.end()), .Width = 1, .Height = 1},
        .ColorSpace = TextureColorSpace::Srgb,
        .WrapS = vk::SamplerAddressMode::eRepeat,
        .WrapT = vk::SamplerAddressMode::eRepeat,
        .Sampler = SamplerConfig{},
        .Name = "DefaultWhite",
    });

    return viewport;
}

void TearDownStoreCtx(entt::registry &r) {
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseSamplerSlots(slots, CollectSamplerSlots(textures.Textures));

    // Tear down GPU-resource owners before GpuBuffers, since they retire allocations into
    // GpuBuffers.Ctx.Retired (cleared on ~BufferContext, which destroys the VMA allocator).
    r.ctx().erase<EnvironmentStore>();
    r.ctx().erase<TextureStore>();
    r.ctx().erase<MeshStore>();
    r.ctx().erase<ColliderShapeBuffers>();
    r.ctx().erase<GpuBuffers>(); // drops BufferContext, whose dtor destroys the VMA allocator
    r.ctx().erase<OneShotGpu>(); // vk pool/fence (Device in VulkanResources still alive)
    r.ctx().erase<MaterialStore>();
    r.ctx().erase<ObjectIdCounter>();
    r.ctx().erase<NameRegistry>();
    r.ctx().erase<DescriptorSlots>();
    r.ctx().erase<VulkanResources>();
}
