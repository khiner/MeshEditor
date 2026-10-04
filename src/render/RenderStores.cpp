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

void RegisterRenderStoreHandlers(state::Scene &r) {
    r.on_destroy<ModelsBuffer, [](state::Scene &r, state::Entity e) {
        r.Context.emplace<PendingSlotRemovals>().Retired.push_back(e);
    }>();
    r.on_destroy<ArmaturePoseState, [](state::Scene &r, state::Entity e) {
        r.Context.emplace<PendingObjectRemovals>().DeformRanges.append_range(r.get<const ArmaturePoseState>(e).GpuDeformRanges);
    }>();
    // History restores the tracked allocator, so a restore releases nothing.
    r.on_destroy<MorphWeightRange, [](state::Scene &r, state::Entity e) {
        if (!r.Restoring) r.Context.emplace<PendingObjectRemovals>().MorphRanges.push_back(r.get<const MorphWeightRange>(e).Weights);
    }>();
    r.on_destroy<RenderInstance, [](state::Scene &r, state::Entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (ri.BufferIndex == UINT32_MAX) return;
        r.Context.emplace<PendingSlotRemovals>().Instances.push_back({ri.Entity, ri.BufferIndex});
    }>();
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
