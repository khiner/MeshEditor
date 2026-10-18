#pragma once
#include "state/Entity.h"
namespace mtl {
struct Context;
struct BufferContext;
} // namespace mtl
void InitRenderStoreContext(state::Registry &, const mtl::Context &);
void RegisterRenderStoreHandlers(state::Registry &);
mtl::BufferContext &InitRenderStores(state::Registry &);
void InitDefaultMaterial(state::Registry &);
void DeinitTextureStores(state::Registry &);
void DeinitRenderStores(state::Registry &);
void DeinitRenderStoreContext(state::Registry &);
