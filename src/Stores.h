#pragma once

#include <entt/entity/fwd.hpp>

namespace mtl {
struct Context;
} // namespace mtl

// Create registry stores and allocate the white-texture sampler slot.
void InitStoreCtx(entt::registry &, const mtl::Context &);

// Require InitStoreCtx to run first.
entt::entity WireRegistry(entt::registry &);

void TearDownStoreCtx(entt::registry &);
