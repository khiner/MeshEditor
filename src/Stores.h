#pragma once

#include <entt/entity/fwd.hpp>

namespace mtl {
struct Context;
} // namespace mtl

// Create the registry stores and allocate TextureStore's white-texture sampler slot.
void InitStoreCtx(entt::registry &, const mtl::Context &);

// InitStoreCtx must run first.
entt::entity WireRegistry(entt::registry &);

void TearDownStoreCtx(entt::registry &);
