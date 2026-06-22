#pragma once

#include "vulkan/VulkanResources.h"

#include <entt/entity/fwd.hpp>

// Creates the per-registry ctx singleton stores: DescriptorSlots, TextureStore, EnvironmentStore.
// Allocates the white-texture sampler slot on TextureStore.
void InitStoreCtx(entt::registry &, VulkanResources);

// InitStoreCtx must run first.
entt::entity WireRegistry(entt::registry &);

void TearDownStoreCtx(entt::registry &);
