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
struct Mesh;

// Lightweight GPU-storage views for consumers that do not need the full GpuBuffers definition.
std::span<PBRMaterial> GetMaterials(entt::registry &);
// A mesh's triangulated face indices. A triangle mesh's draws index the store's corner array, so the
// corners are its indices. An n-gon mesh fans into a triangulated buffer in the index arena.
std::span<const uint32_t> GetFaceIndices(const entt::registry &, const Mesh &, const MeshBuffers &);
std::span<const PunctualLight> GetLights(entt::registry &);
PunctualLight GetLight(entt::registry &, uint32_t index);
mtl::BufferContext &GetBufferContext(entt::registry &);
void ReleaseMeshBuffers(entt::registry &, MeshBuffers &);

void FreeInstanceRange(entt::registry &, Range);
void ReleaseEdgeIndices(entt::registry &, const SlottedRange &);
