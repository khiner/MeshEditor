#include "Stores.h"
#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "audio/AudioStores.h"
#include "mesh/MeshStores.h"
#include "physics/PhysicsStores.h"
#include "render/RenderStores.h"
#include "scene/Entity.h"
#include <entt/entity/registry.hpp>

void InitStoreCtx(entt::registry &r, const mtl::Context &ctx) { InitRenderStoreContext(r, ctx); }

entt::entity WireRegistry(entt::registry &r) {
    RegisterMeshStoreHandlers(r);
    RegisterAudioStoreHandlers(r);
    RegisterPhysicsStoreHandlers(r);
    InitEntityNames(r);
    RegisterRenderStoreHandlers(r);

    const auto viewport = r.create();
    auto &buffers = InitRenderStores(r);
    InitMeshStore(r, buffers);
    r.ctx().emplace<action::Errors>();
    r.emplace<TimelineRange>(viewport);
    r.emplace<TimelinePlayback>(viewport);
    InitDefaultMaterial(r, viewport);
    return viewport;
}

void TearDownStoreCtx(entt::registry &r) {
    // MeshHandle destruction needs the mesh store; resource owners retire buffers into the render store.
    ClearMeshStoreHandles(r);
    DeinitTextureStores(r);
    DeinitMeshStore(r);
    DeinitRenderStores(r);
    DeinitEntityNames(r);
    DeinitRenderStoreContext(r);
}
