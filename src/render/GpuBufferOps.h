#pragma once

#include "Range.h"
#include "SlottedRange.h"
#include "gpu/PBRMaterial.h"

#include <span>

#include "state/Entity.h"

namespace mtl {
struct BufferContext;
} // namespace mtl
struct MeshBuffers;
struct Mesh;

std::span<const PBRMaterial> GetMaterials(const state::Scene &);
// Returns store corners for triangle meshes or the triangulated index-arena range for n-gons.
std::span<const uint32_t> GetFaceIndices(const state::Scene &, const Mesh &);
mtl::BufferContext &GetBufferContext(state::Scene &);
// The render ranges of the record an entity draws. TryMeshBuffers is null before the record's first sync.
const MeshBuffers *TryMeshBuffers(const state::Scene &, state::Entity);
const MeshBuffers &MeshBuffersOf(const state::Scene &, state::Entity);
MeshBuffers &MeshBuffersOf(state::Scene &, state::Entity);

void FreeInstanceRange(state::Scene &, Range);
void ReleaseEdgeIndices(state::Scene &, const SlottedRange &);
