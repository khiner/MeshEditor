#include "render/RenderStores.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/Bindless.h"
#include "object/PendingSync.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "state/Scene.h"
#include "viewport/ViewportDisplay.h"

void InitRenderStoreContext(state::Scene &r, const mtl::Context &ctx) {
    auto &slots = r.ctx().emplace<mtl::BindlessSet>(ctx);
    r.ctx().emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(ViewportDisplay{}.AnisotropicFilter)));
    auto &textures = r.ctx().emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.ctx().emplace<EnvironmentStore>();
}

namespace {
template<typename Handle>
void EmplaceMeshBuffers(state::Scene &r, state::Entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    r.emplace<MeshBuffers>(e, meshes.Arenas().Vertices.Slotted(meshes.Get(r.get<const Handle>(e).StoreId).Vertices), SlottedRange{}, SlottedRange{}, SlottedRange{});
}

void EmplaceMeshShadingSummary(state::Scene &r, state::Entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto [any, all] = meshes.GetFaceSharpnessSummary(r.get<const MeshHandle>(e).StoreId);
    r.emplace_or_replace<MeshShadingSummary>(e, any, all);
}
} // namespace
void RegisterRenderStoreHandlers(state::Scene &r) {
    r.on_destroy<ArmaturePoseState>().connect<[](state::Scene &r, state::Entity e) {
        auto &buffer = r.ctx().get<GpuBuffers>().ArmatureDeformBuffer;
        for (const auto range : r.get<const ArmaturePoseState>(e).GpuDeformRanges) buffer.Release(range);
    }>();
    // History restores the tracked allocator, so a restore releases nothing.
    r.on_destroy<MorphWeightRange>().connect<[](state::Scene &r, state::Entity e) {
        if (!r.Restoring) r.ctx().get<GpuBuffers>().MorphWeightBuffer.Release(r.get<const MorphWeightRange>(e).Weights);
    }>();
    r.on_destroy<MeshHandle>().connect<&state::Scene::remove<MeshShadingSummary>>();
    r.on_destroy<RenderInstance>().connect<[](state::Scene &r, state::Entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (auto *buffers = r.ctx().find<GpuBuffers>()) {
            buffers->MeshletRangeCount -= ri.MeshletRangeCount;
            buffers->MeshletInstanceCount -= ri.MeshletCount;
        }
        if (ri.BufferIndex == UINT32_MAX) return;
        r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
    }>();
    // Keep RenderInstance synchronized with Instance and Hidden regardless of snapshot insertion order.
    r.on_construct<Instance>().connect<[](state::Scene &r, state::Entity e) {
        if (!r.all_of<Hidden>(e) && !r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX);
    }>();
    r.on_construct<Hidden>().connect<[](state::Scene &r, state::Entity e) {
        if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
    }>();
    r.on_construct<MeshHandle>().connect<&EmplaceMeshBuffers<MeshHandle>>();
    r.on_construct<MeshHandle>().connect<&EmplaceMeshShadingSummary>();
    r.on_construct<VertexStoreId>().connect<&EmplaceMeshBuffers<VertexStoreId>>();
}
mtl::BufferContext &InitRenderStores(state::Scene &r) {
    auto &buffers = r.ctx().emplace<GpuBuffers>(r.ctx().get<const mtl::Context>(), r.ctx().get<mtl::BindlessSet>());
    r.ctx().emplace<MaterialStore>();
    return buffers.Ctx;
}
void InitDefaultMaterial(state::Scene &r) {
    auto &buffers = r.ctx().get<GpuBuffers>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &materials = r.ctx().get<MaterialStore>();
    buffers.Materials.Append(PBRMaterial{.MetallicFactor = 0.f, .BaseColorTexture = {.Slot = textures.WhiteTextureSlot}});
    materials.AppendNames({"Default"});

    constexpr std::array<std::byte, 4> WhitePixels{std::byte{0xff}, std::byte{0xff}, std::byte{0xff}, std::byte{0xff}};
    textures.PendingUploads.emplace_back(PendingTextureUpload{
        .SamplerSlot = textures.WhiteTextureSlot,
        .Source = PendingTextureUpload::RawPixels{.Pixels = std::vector<std::byte>(WhitePixels.begin(), WhitePixels.end()), .Width = 1, .Height = 1},
        .Params = {
            .ColorSpace = TextureColorSpace::Srgb,
            .WrapS = MTL::SamplerAddressModeRepeat,
            .WrapT = MTL::SamplerAddressModeRepeat,
            .Sampler = SamplerConfig{},
            .Name = "DefaultWhite",
        },
    });
}
void DeinitTextureStores(state::Scene &r) {
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseTextureSlots(slots, textures.Textures);
    r.ctx().erase<EnvironmentStore>();
    r.ctx().erase<TextureStore>();
}
void DeinitRenderStores(state::Scene &r) {
    r.ctx().erase<GpuBuffers>();
    r.ctx().erase<MaterialStore>();
}
void DeinitRenderStoreContext(state::Scene &r) { r.ctx().erase<mtl::BindlessSet>(); }
