#include "render/RenderStores.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "object/PendingSync.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "state/Registry.h"
#include "viewport/ViewportDisplay.h"

void InitRenderStoreContext(state::Registry &r, const mtl::Context &ctx) {
    auto &slots = r.Resources.emplace<mtl::BindlessSet>(ctx);
    r.Resources.emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(AnisotropicFilter{}.Value)));
    auto &textures = r.Resources.emplace<TextureStore>();
    textures.WhiteTextureSlot = AllocateSamplerSlot(slots);
    r.Resources.emplace<EnvironmentStore>();
}

void RegisterRenderStoreHandlers(state::Registry &r) {
    r.on_destroy<ModelsBuffer, [](state::Registry &r, state::Entity e) {
        r.Resources.emplace<PendingSlotRemovals>().Retired.push_back(e);
    }>();
    r.on_destroy<ArmaturePoseState, [](state::Registry &r, state::Entity e) {
        r.Resources.emplace<PendingObjectRemovals>().DeformRanges.append_range(r.get<const ArmaturePoseState>(e).GpuDeformRanges);
    }>();
    // History restores the tracked allocator, so a restore releases nothing.
    r.on_destroy<MorphWeightRange, [](state::Registry &r, state::Entity e) {
        if (!r.Restoring) r.Resources.emplace<PendingObjectRemovals>().MorphRanges.push_back(r.get<const MorphWeightRange>(e).Weights);
    }>();
    r.on_destroy<RenderInstance, [](state::Registry &r, state::Entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (ri.BufferIndex == UINT32_MAX) return;
        r.Resources.emplace<PendingSlotRemovals>().Instances.push_back({ri.Entity, ri.BufferIndex});
    }>();
}
mtl::BufferContext &InitRenderStores(state::Registry &r) {
    auto &buffers = r.Resources.emplace<GpuBuffers>(r.Resources.get<const mtl::Context>(), r.Resources.get<mtl::BindlessSet>());
    r.Resources.emplace<MaterialStore>();
    return buffers.Ctx;
}
void InitDefaultMaterial(state::Registry &r) {
    auto &buffers = r.Resources.get<GpuBuffers>();
    auto &textures = r.Resources.get<TextureStore>();
    auto &materials = r.Resources.get<MaterialStore>();
    buffers.Materials.Append(PBRMaterial{.BaseColorTexture = {.Slot = textures.WhiteTextureSlot}, .MetallicFactor = 0.f});
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
void DeinitTextureStores(state::Registry &r) {
    auto &slots = r.Resources.get<mtl::BindlessSet>();
    auto &textures = r.Resources.get<TextureStore>();
    const auto &environments = r.Resources.get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseTextureSlots(slots, textures.Textures);
    r.Resources.erase<EnvironmentStore>();
    r.Resources.erase<TextureStore>();
}
void DeinitRenderStores(state::Registry &r) {
    r.Resources.erase<GpuBuffers>();
    r.Resources.erase<MaterialStore>();
}
void DeinitRenderStoreContext(state::Registry &r) { r.Resources.erase<mtl::BindlessSet>(); }
