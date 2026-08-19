#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"

#include <span>

#include <entt/entity/fwd.hpp>

namespace mtl {
struct BufferContext;
} // namespace mtl
struct MeshBuffers;

// Lightweight GPU-storage views for consumers that do not need the full GpuBuffers definition.
std::span<PBRMaterial> GetMaterials(entt::registry &);
// A mesh's triangulated face indices, viewing the index arena's own storage.
std::span<const uint32_t> GetFaceIndices(const entt::registry &, const MeshBuffers &);
std::span<const PunctualLight> GetLights(entt::registry &);
PunctualLight GetLight(entt::registry &, uint32_t index);
mtl::BufferContext &GetBufferContext(entt::registry &);
void ReleaseMeshBuffers(entt::registry &, MeshBuffers &);

void ReleaseVertexClasses(entt::registry &, uint32_t offset, uint32_t count);
void FreeInstanceRange(entt::registry &, Range);
void ReleaseEdgeIndices(entt::registry &, const SlottedRange &);
