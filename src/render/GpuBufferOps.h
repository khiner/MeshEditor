#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"

#include <span>

#include <entt/entity/fwd.hpp>

namespace mvk {
struct BufferContext;
}
struct MeshBuffers;

// Vulkan-free views and ops over GpuBuffers' GPU storage, so read/UI/action consumers
// need not include the full (vulkan-heavy) GpuBuffers header.
std::span<PBRMaterial> GetMaterials(entt::registry &);
std::span<const PunctualLight> GetLights(entt::registry &);
PunctualLight GetLight(entt::registry &, uint32_t index);
mvk::BufferContext &GetBufferContext(entt::registry &);
void ReleaseMeshBuffers(entt::registry &, MeshBuffers &);

// Vulkan-free buffer mutations.
uint32_t AllocateVertexClasses(entt::registry &, std::span<const uint8_t>);
void ReleaseVertexClasses(entt::registry &, uint32_t offset, uint32_t count);
void FreeInstanceRange(entt::registry &, Range);
void ReleaseEdgeIndices(entt::registry &, const SlottedRange &);
