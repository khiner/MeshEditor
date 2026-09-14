#pragma once

#include "state/Entity.h"

namespace mtl {
struct Context;
} // namespace mtl

// Create registry stores and allocate the white-texture sampler slot.
void InitStoreCtx(state::Scene &, const mtl::Context &);

// Require InitStoreCtx to run first.
state::Entity InitDocumentStores(state::Scene &);

void TearDownStoreCtx(state::Scene &);
