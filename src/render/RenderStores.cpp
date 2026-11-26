#include "render/RenderStores.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStores.h"
#include "object/PendingSync.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "state/Registry.h"
#include "viewport/ViewportDisplay.h"

void InitRenderStoreContext(state::Registry &r, const mtl::Context &ctx) {
    r.Resources.emplace<mtl::BindlessSet>(ctx);
    r.Resources.emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(AnisotropicFilter{}.Value)));
    r.Resources.emplace<TextureStore>();
    r.Resources.emplace<EnvironmentStore>();
}

void RegisterRenderStoreHandlers(state::Registry &r) {
    r.on_destroy<ModelsBuffer, [](state::Registry &r, state::Entity e) {
        r.Resources.emplace<PendingSlotRemovals>().Retired.push_back(e);
        if (!r.Restoring) r.Resources.emplace<PendingObjectRemovals>().InstanceRanges.push_back(r.get<const ModelsBuffer>(e).InstanceRange);
    }>();
    r.on_destroy<ArmaturePoseState, [](state::Registry &r, state::Entity e) {
        r.Resources.emplace<PendingObjectRemovals>().DeformRanges.append_range(r.get<const ArmaturePoseState>(e).GpuDeformRanges);
    }>();
    // History restores the tracked allocator, so a restore releases nothing.
    r.on_destroy<MorphWeightRange, [](state::Registry &r, state::Entity e) {
        if (!r.Restoring) r.Resources.emplace<PendingObjectRemovals>().MorphRanges.push_back(r.get<const MorphWeightRange>(e).Weights);
    }>();
    // A destroyed mesh entity's record drops its tallies unless another entity draws the record.
    r.on_destroy<TalliedRecord, [](state::Registry &r, state::Entity e) {
        const auto store_id = r.get<const TalliedRecord>(e).StoreId;
        if (const auto owner = MeshEntityOf(r, store_id); owner == state::Null || owner == e) r.Resources.get<GpuBuffers>().Retally(store_id, {});
    }>();
    r.on_destroy<RenderInstance, [](state::Registry &r, state::Entity e) {
        const auto &ri = r.get<const RenderInstance>(e);
        if (ri.BufferIndex == UINT32_MAX) return;
        r.Resources.emplace<PendingSlotRemovals>().Instances.push_back({ri.Entity, ri.BufferIndex});
    }>();
}
mtl::BufferContext &InitRenderStores(state::Registry &r) {
    auto &buffers = r.Resources.emplace<GpuBuffers>(r.Resources.get<const mtl::Context>(), r.Resources.get<mtl::BindlessSet>());
    return buffers.Ctx;
}
void InitDefaultMaterial(state::Registry &r) {
    r.Resources.get<GpuBuffers>().Materials.Append(PBRMaterial{.MetallicFactor = 0.f, .DoubleSided = 1u});
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
}
void DeinitRenderStoreContext(state::Registry &r) { r.Resources.erase<mtl::BindlessSet>(); }
