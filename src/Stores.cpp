#include "Stores.h"

#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "object/ExtrasComponents.h"
#include "object/ObjectComponents.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "render/Bindless.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/OneShotGpu.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

void InitStoreCtx(entt::registry &r, VulkanResources vk) {
    vk.MaxSamplerAnisotropy = ClampMaxAnisotropy(vk, ToMaxAnisotropy(ViewportDisplay{}.AnisotropicFilter));
    r.ctx().emplace<VulkanResources>(vk);
    auto &slots = r.ctx().emplace<DescriptorSlots>(
        vk.Device,
        vk.PhysicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDescriptorIndexingProperties>().get<vk::PhysicalDeviceDescriptorIndexingProperties>()
    );
    auto &textures = r.ctx().emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.ctx().emplace<EnvironmentStore>();
}

namespace {
// on_construct hook: default-create C, but only if absent, so a snapshot-restored C isn't clobbered or double-emplaced.
template<typename C>
void EmplaceIfAbsent(entt::registry &r, entt::entity e) {
    if (!r.all_of<C>(e)) r.emplace<C>(e);
}

// on_construct hook: build the entity's MeshBuffers from its store handle's vertex range. Index ranges fill in afterward.
template<typename Handle, auto GetRange>
void EmplaceMeshBuffers(entt::registry &r, entt::entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    r.emplace<MeshBuffers>(e, (meshes.*GetRange)(r.get<const Handle>(e).StoreId), SlottedRange{}, SlottedRange{}, SlottedRange{});
}
} // namespace

entt::entity WireRegistry(entt::registry &r) {
    r.on_destroy<MeshHandle>().connect<[](entt::registry &r, entt::entity e) {
        r.ctx().get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
    r.on_construct<PhysicsMotion>().connect<&EmplaceIfAbsent<PhysicsVelocity>>();
    r.on_destroy<PhysicsMotion>().connect<&entt::registry::remove<PhysicsVelocity>>();
    r.on_construct<ColliderShape>().connect<&EmplaceIfAbsent<ColliderMaterial>>();
    r.on_destroy<ColliderShape>().connect<&entt::registry::remove<ColliderMaterial>>();

    r.on_destroy<Name>().connect<[](entt::registry &r, entt::entity e) {
        if (auto *registry = r.ctx().find<NameRegistry>()) registry->Names.erase(r.get<const Name>(e).Value);
    }>();
    // Assign a stable ObjectId (0 means unassigned) on RenderInstance construction.
    r.on_construct<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        if (r.get<const RenderInstance>(e).ObjectId != 0) return;
        if (auto *counter = r.ctx().find<ObjectIdCounter>()) {
            r.patch<RenderInstance>(e, [counter](auto &ri) { ri.ObjectId = counter->Next++; });
        }
    }>();
    r.on_destroy<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (ri.BufferIndex == UINT32_MAX) return; // Same-frame show+hide — never synced to GPU.
        r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
    }>();
    // An instance renders unless Hidden: create its RenderInstance on construction, drop it when Hidden appears.
    // Together these keep RenderInstance in lockstep with Instance + !Hidden, including on snapshot restore
    // (which emplaces Instance and Hidden in either order).
    r.on_construct<Instance>().connect<[](entt::registry &r, entt::entity e) {
        if (!r.all_of<Hidden>(e) && !r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX, 0u);
    }>();
    r.on_construct<Hidden>().connect<[](entt::registry &r, entt::entity e) {
        if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
    }>();
    // Build MeshBuffers when a vertex handle is constructed (MeshHandle = full meshes,
    // VertexStoreId = vertex-only extras, OverlayVertexStoreId = overlays).
    r.on_construct<MeshHandle>().connect<&EmplaceMeshBuffers<MeshHandle, &MeshStore::GetVerticesRange>>();
    r.on_construct<VertexStoreId>().connect<&EmplaceMeshBuffers<VertexStoreId, &MeshStore::GetVerticesRange>>();
    r.on_construct<OverlayVertexStoreId>().connect<&EmplaceMeshBuffers<OverlayVertexStoreId, &MeshStore::GetOverlayVerticesRange>>();

    const auto &vk = r.ctx().get<const VulkanResources>();
    auto &slots = r.ctx().get<DescriptorSlots>();
    auto &textures = r.ctx().get<TextureStore>();

    const auto viewport = r.create();
    auto &buffers = r.ctx().emplace<GpuBuffers>(vk.PhysicalDevice, vk.Device, vk.Instance, slots);
    r.ctx().emplace<MeshStore>(buffers.Ctx);

    r.ctx().emplace<NameRegistry>();
    r.ctx().emplace<ObjectIdCounter>();
    r.ctx().emplace<ColliderShapeBuffers>();
    r.ctx().emplace<action::Errors>();
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
    // Releasing a MeshHandle calls back into MeshStore, so clear handles while the store is still alive.
    r.clear<MeshHandle>();

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
