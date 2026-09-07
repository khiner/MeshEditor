#include "render/RenderStores.h"
#include "animation/AnimationTimeline.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/Bindless.h"
#include "object/ObjectComponents.h"
#include "object/PendingSync.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

void InitRenderStoreContext(entt::registry &r, const mtl::Context &ctx) {
    auto &slots = r.ctx().emplace<mtl::BindlessSet>(ctx);
    r.ctx().emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(ViewportDisplay{}.AnisotropicFilter)));
    auto &textures = r.ctx().emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.ctx().emplace<EnvironmentStore>();
}

namespace {
template<typename Handle>
void EmplaceMeshBuffers(entt::registry &r, entt::entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    r.emplace<MeshBuffers>(e, meshes.GetVerticesRange(r.get<const Handle>(e).StoreId), SlottedRange{}, SlottedRange{}, SlottedRange{});
}

void EmplaceMeshShadingSummary(entt::registry &r, entt::entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto [any, all] = meshes.GetFaceSharpnessSummary(r.get<const MeshHandle>(e).StoreId);
    r.emplace_or_replace<MeshShadingSummary>(e, any, all);
}
} // namespace
void RegisterRenderStoreHandlers(entt::registry &r) {
    r.on_destroy<MeshHandle>().connect<&entt::registry::remove<MeshShadingSummary>>();
    // Assign stable nonzero object identifiers to new render instances.
    r.on_construct<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        if (r.get<const RenderInstance>(e).ObjectId != 0) return;
        if (auto *counter = r.ctx().find<ObjectIdCounter>()) {
            r.patch<RenderInstance>(e, [counter](auto &ri) { ri.ObjectId = counter->Next++; });
        }
    }>();
    r.on_destroy<RenderInstance>().connect<[](entt::registry &r, entt::entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (auto *buffers = r.ctx().find<GpuBuffers>()) {
            buffers->MeshletRangeCount -= ri.MeshletRangeCount;
            buffers->MeshletInstanceCount -= ri.MeshletCount;
        }
        if (ri.BufferIndex == UINT32_MAX) return;
        r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
    }>();
    // Keep RenderInstance synchronized with Instance and Hidden regardless of snapshot insertion order.
    r.on_construct<Instance>().connect<[](entt::registry &r, entt::entity e) {
        if (!r.all_of<Hidden>(e) && !r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX, 0u);
    }>();
    r.on_construct<Hidden>().connect<[](entt::registry &r, entt::entity e) {
        if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
    }>();
    r.on_construct<MeshHandle>().connect<&EmplaceMeshBuffers<MeshHandle>>();
    r.on_construct<MeshHandle>().connect<&EmplaceMeshShadingSummary>();
    r.on_construct<VertexStoreId>().connect<&EmplaceMeshBuffers<VertexStoreId>>();
}
mtl::BufferContext &InitRenderStores(entt::registry &r) {
    auto &buffers = r.ctx().emplace<GpuBuffers>(r.ctx().get<const mtl::Context>(), r.ctx().get<mtl::BindlessSet>());
    r.ctx().emplace<ObjectIdCounter>();
    r.ctx().emplace<MaterialStore>();
    return buffers.Ctx;
}
void InitDefaultMaterial(entt::registry &r, entt::entity viewport) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &materials = r.ctx().get<MaterialStore>();
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
        .WrapS = MTL::SamplerAddressModeRepeat,
        .WrapT = MTL::SamplerAddressModeRepeat,
        .Sampler = SamplerConfig{},
        .Name = "DefaultWhite",
    });
}
void DeinitTextureStores(entt::registry &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseSamplerSlots(slots, CollectSamplerSlots(textures.Textures));
    r.ctx().erase<EnvironmentStore>();
    r.ctx().erase<TextureStore>();
}
void DeinitRenderStores(entt::registry &r) {
    r.ctx().erase<GpuBuffers>();
    r.ctx().erase<MaterialStore>();
    r.ctx().erase<ObjectIdCounter>();
}
void DeinitRenderStoreContext(entt::registry &r) { r.ctx().erase<mtl::BindlessSet>(); }
