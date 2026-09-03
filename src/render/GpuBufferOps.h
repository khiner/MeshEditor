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

std::span<PBRMaterial> GetMaterials(entt::registry &);
// Returns store corners for triangle meshes or the triangulated index-arena range for n-gons.
std::span<const uint32_t> GetFaceIndices(const entt::registry &, const Mesh &, const MeshBuffers &);
std::span<const PunctualLight> GetLights(entt::registry &);
PunctualLight GetLight(entt::registry &, uint32_t index);
mtl::BufferContext &GetBufferContext(entt::registry &);
void ReleaseMeshBuffers(entt::registry &, MeshBuffers &);

void FreeInstanceRange(entt::registry &, Range);
void ReleaseEdgeIndices(entt::registry &, const SlottedRange &);
