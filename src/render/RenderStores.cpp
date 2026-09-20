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
    auto &slots = r.Context.emplace<mtl::BindlessSet>(ctx);
    r.Context.emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(ViewportDisplay{}.AnisotropicFilter)));
    auto &textures = r.Context.emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.Context.emplace<EnvironmentStore>();
}

namespace {
void EmplaceMeshShadingSummary(state::Scene &r, state::Entity e) {
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto [any, all] = meshes.GetFaceSharpnessSummary(GetMesh(r, e).GetStoreId());
    r.emplace_or_replace<MeshShadingSummary>(e, any, all);
}
} // namespace
void RegisterRenderStoreHandlers(state::Scene &r) {
    r.on_destroy<ArmaturePoseState, [](state::Scene &r, state::Entity e) {
        auto &buffer = r.Context.get<GpuBuffers>().ArmatureDeformBuffer;
        for (const auto range : r.get<const ArmaturePoseState>(e).GpuDeformRanges) buffer.Release(range);
    }>();
    // History restores the tracked allocator, so a restore releases nothing.
    r.on_destroy<MorphWeightRange, [](state::Scene &r, state::Entity e) {
        if (!r.Restoring) r.Context.get<GpuBuffers>().MorphWeightBuffer.Release(r.get<const MorphWeightRange>(e).Weights);
    }>();
    r.on_destroy<MeshHandle, &state::Scene::remove<MeshShadingSummary>>();
    r.on_destroy<RenderInstance, [](state::Scene &r, state::Entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (auto *buffers = r.Context.find<GpuBuffers>()) {
            buffers->MeshletRangeCount -= ri.MeshletRangeCount;
            buffers->MeshletInstanceCount -= ri.MeshletCount;
        }
        if (ri.BufferIndex == UINT32_MAX) return;
        r.get_or_emplace<PendingHide>(ri.Entity).BufferIndices.push_back(ri.BufferIndex);
    }>();
    // Keep RenderInstance synchronized with Instance and Hidden regardless of snapshot insertion order.
    r.on_construct<Instance, [](state::Scene &r, state::Entity e) {
        if (!r.all_of<Hidden>(e) && !r.all_of<RenderInstance>(e)) r.emplace<RenderInstance>(e, r.get<Instance>(e).Entity, UINT32_MAX);
    }>();
    r.on_construct<Hidden, [](state::Scene &r, state::Entity e) {
        if (r.all_of<RenderInstance>(e)) r.remove<RenderInstance>(e);
    }>();
    r.on_construct<MeshHandle, &EmplaceMeshShadingSummary>();
}
mtl::BufferContext &InitRenderStores(state::Scene &r) {
    auto &buffers = r.Context.emplace<GpuBuffers>(r.Context.get<const mtl::Context>(), r.Context.get<mtl::BindlessSet>());
    r.Context.emplace<MaterialStore>();
    return buffers.Ctx;
}
void InitDefaultMaterial(state::Scene &r) {
    auto &buffers = r.Context.get<GpuBuffers>();
    auto &textures = r.Context.get<TextureStore>();
    auto &materials = r.Context.get<MaterialStore>();
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
    auto &slots = r.Context.get<mtl::BindlessSet>();
    auto &textures = r.Context.get<TextureStore>();
    auto &environments = r.Context.get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseTextureSlots(slots, textures.Textures);
    r.Context.erase<EnvironmentStore>();
    r.Context.erase<TextureStore>();
}
void DeinitRenderStores(state::Scene &r) {
    r.Context.erase<GpuBuffers>();
    r.Context.erase<MaterialStore>();
}
void DeinitRenderStoreContext(state::Scene &r) { r.Context.erase<mtl::BindlessSet>(); }
