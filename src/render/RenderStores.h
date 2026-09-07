#pragma once
#include "entt_fwd.h"
namespace mtl {
struct Context;
struct BufferContext;
} // namespace mtl
void InitRenderStoreContext(entt::registry &, const mtl::Context &);
void RegisterRenderStoreHandlers(entt::registry &);
mtl::BufferContext &InitRenderStores(entt::registry &);
void InitDefaultMaterial(entt::registry &, entt::entity);
void DeinitTextureStores(entt::registry &);
void DeinitRenderStores(entt::registry &);
void DeinitRenderStoreContext(entt::registry &);
