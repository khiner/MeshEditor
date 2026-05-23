#pragma once

#include "VulkanResources.h"

#include <entt/entity/fwd.hpp>

// Creates the per-registry scene-singleton stores: DescriptorSlots, TextureStore, EnvironmentStore in ctx.
// Allocates the white-texture sampler slot on TextureStore.
void InitSceneStoreCtx(entt::registry &, VulkanResources);

// Connects component-lifecycle hooks, creates the scene entity with its GpuBuffers component,
// emplaces MeshStore in ctx (depends on GpuBuffers.Ctx), seeds the default material + white texture upload,
// and returns the scene entity. Requires InitSceneStoreCtx to have run.
entt::entity WireSceneRegistry(entt::registry &);

// Releases sampler slots held by TextureStore and EnvironmentStore, then erases the ctx singletons
// (MeshStore, EnvironmentStore, TextureStore, DescriptorSlots) and removes GpuBuffers from the scene entity.
void TearDownSceneStoreCtx(entt::registry &);
