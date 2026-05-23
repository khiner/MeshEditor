#pragma once

#include "VulkanResources.h"

#include <entt/entity/fwd.hpp>

// Creates the per-registry ctx singleton stores: DescriptorSlots, TextureStore, EnvironmentStore.
// Allocates the white-texture sampler slot on TextureStore.
void InitStoreCtx(entt::registry &, VulkanResources);

// Connects component-lifecycle hooks, creates the viewport entity, emplaces GpuBuffers and
// MeshStore in ctx (MeshStore depends on GpuBuffers.Ctx), seeds the default material + white
// texture upload, and returns the viewport entity. Requires InitStoreCtx to have run.
entt::entity WireRegistry(entt::registry &);

// Releases sampler slots held by TextureStore and EnvironmentStore, then erases the ctx singletons
// (MeshStore, EnvironmentStore, TextureStore, GpuBuffers, DescriptorSlots).
void TearDownStoreCtx(entt::registry &);
