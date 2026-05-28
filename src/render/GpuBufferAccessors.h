#pragma once

#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"
#include "gpu/WorkspaceLights.h"

#include <entt/entity/fwd.hpp>

#include <span>

namespace mvk {
struct BufferContext;
}
struct MeshBuffers;

// Vulkan-free views and ops over GpuBuffers' GPU storage, so read/UI/action consumers
// need not include the full (vulkan-heavy) GpuBuffers header.
std::span<PBRMaterial> GetMaterials(entt::registry &);
std::span<const PunctualLight> GetLights(entt::registry &);
WorkspaceLights &GetWorkspaceLights(entt::registry &);
PunctualLight GetLight(entt::registry &, uint32_t index);
mvk::BufferContext &GetBufferContext(entt::registry &);
void ReleaseMeshBuffers(entt::registry &, MeshBuffers &);
