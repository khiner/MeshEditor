#pragma once
#include "state/Entity.h"
namespace mtl {
struct Context;
struct BufferContext;
} // namespace mtl
void InitRenderStoreContext(state::Scene &, const mtl::Context &);
void RegisterRenderStoreHandlers(state::Scene &);
mtl::BufferContext &InitRenderStores(state::Scene &);
void InitDefaultMaterial(state::Scene &);
void DeinitTextureStores(state::Scene &);
void DeinitRenderStores(state::Scene &);
void DeinitRenderStoreContext(state::Scene &);
