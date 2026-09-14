#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/PBRMaterial.h"
#include "gpu/PunctualLight.h"

#include <span>

#include "state/Entity.h"

namespace mtl {
struct BufferContext;
} // namespace mtl
struct MeshBuffers;
struct Mesh;

std::span<const PBRMaterial> GetMaterials(const state::Scene &);
// Returns store corners for triangle meshes or the triangulated index-arena range for n-gons.
std::span<const uint32_t> GetFaceIndices(const state::Scene &, const Mesh &, const MeshBuffers &);
std::span<const PunctualLight> GetLights(state::Scene &);
PunctualLight GetLight(state::Scene &, uint32_t index);
mtl::BufferContext &GetBufferContext(state::Scene &);
void ReleaseMeshBuffers(state::Scene &, MeshBuffers &);

void FreeInstanceRange(state::Scene &, Range);
void ReleaseEdgeIndices(state::Scene &, const SlottedRange &);
