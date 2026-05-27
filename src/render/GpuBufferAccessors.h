#pragma once

#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/WorkspaceLights.h"

#include <entt/entity/fwd.hpp>

#include <span>

// Vulkan-free views into GpuBuffers' GPU-mapped storage, so read/UI consumers
// need not include the full (vulkan-heavy) GpuBuffers header.
std::span<PBRMaterial> GetMaterials(entt::registry &);
std::span<const PunctualLight> GetLights(entt::registry &);
WorkspaceLights &GetWorkspaceLights(entt::registry &);
