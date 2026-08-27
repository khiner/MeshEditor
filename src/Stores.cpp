#include "Stores.h"

#include "audio/SoundVertices.h"

#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "metal/Bindless.h"
#include "object/ObjectComponents.h"
#include "object/PendingSync.h"
#include "physics/PhysicsTypes.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "viewport/ViewportDisplay.h"

#include <entt/entity/registry.hpp>

void InitStoreCtx(entt::registry &r, const mtl::Context &ctx) {
    auto &slots = r.ctx().emplace<mtl::BindlessSet>(ctx);
    r.ctx().emplace<ActiveSamplerAnisotropy>(ClampMaxAnisotropy(ToMaxAnisotropy(ViewportDisplay{}.AnisotropicFilter)));
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

void EmplaceMeshShadingSummary(entt::registry &r, entt::entity e) {
    const auto &meshes = r.ctx().get<const MeshStore>();
    const auto [any, all] = meshes.GetFaceSharpnessSummary(r.get<const MeshHandle>(e).StoreId);
    r.emplace_or_replace<MeshShadingSummary>(e, any, all);
}
} // namespace

entt::entity WireRegistry(entt::registry &r) {
    r.on_destroy<MeshHandle>().connect<[](entt::registry &r, entt::entity e) {
        r.ctx().get<MeshStore>().Release(r.get<MeshHandle>(e).StoreId);
    }>();
    r.on_destroy<MeshHandle>().connect<&entt::registry::remove<MeshShadingSummary>>();
    r.on_destroy<SoundVertices>().connect<[](entt::registry &r, entt::entity e) {
        r.ctx().get<MeshStore>().ReleaseSoundVertices(r.get<SoundVertices>(e).Vertices);
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
        if (auto *buffers = r.ctx().find<GpuBuffers>()) {
            buffers->MeshletRangeCount -= ri.MeshletRangeCount;
            buffers->MeshletInstanceCount -= ri.MeshletCount;
            if (ri.GpuId != InvalidOffset) {
                buffers->GpuInstanceSlots.GetMutable({ri.GpuId, 1})[0] = InvalidOffset;
                buffers->GpuInstanceSlots.Release({ri.GpuId, 1});
            }
        }
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
    // Build MeshBuffers when a vertex handle is constructed (MeshHandle = full meshes, VertexStoreId = vertex-only extras).
    r.on_construct<MeshHandle>().connect<&EmplaceMeshBuffers<MeshHandle, &MeshStore::GetVerticesRange>>();
    r.on_construct<MeshHandle>().connect<&EmplaceMeshShadingSummary>();
    r.on_construct<VertexStoreId>().connect<&EmplaceMeshBuffers<VertexStoreId, &MeshStore::GetVerticesRange>>();

    const auto &ctx = r.ctx().get<const mtl::Context>();
    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &textures = r.ctx().get<TextureStore>();

    const auto viewport = r.create();
    auto &buffers = r.ctx().emplace<GpuBuffers>(ctx, slots);
    r.ctx().emplace<MeshStore>(buffers.Ctx);

    r.ctx().emplace<NameRegistry>();
    r.ctx().emplace<ObjectIdCounter>();
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
        .WrapS = MTL::SamplerAddressModeRepeat,
        .WrapT = MTL::SamplerAddressModeRepeat,
        .Sampler = SamplerConfig{},
        .Name = "DefaultWhite",
    });

    return viewport;
}

void TearDownStoreCtx(entt::registry &r) {
    // Releasing a MeshHandle calls back into MeshStore, so clear handles while the store is still alive.
    r.clear<MeshHandle>();

    auto &slots = r.ctx().get<mtl::BindlessSet>();
    auto &textures = r.ctx().get<TextureStore>();
    auto &environments = r.ctx().get<EnvironmentStore>();
    ReleaseEnvironmentSamplerSlots(slots, environments);
    ReleaseSamplerSlots(slots, CollectSamplerSlots(textures.Textures));

    // Tear down GPU-resource owners before GpuBuffers, since they retire allocations into
    // GpuBuffers.Ctx.Retired, which ~BufferContext clears.
    r.ctx().erase<EnvironmentStore>();
    r.ctx().erase<TextureStore>();
    r.ctx().erase<MeshStore>();
    r.ctx().erase<GpuBuffers>(); // Drops BufferContext, releasing every retired buffer.
    r.ctx().erase<MaterialStore>();
    r.ctx().erase<ObjectIdCounter>();
    r.ctx().erase<NameRegistry>();
    r.ctx().erase<mtl::BindlessSet>();
}
