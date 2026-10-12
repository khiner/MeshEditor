#pragma once
#include "mesh/ElementAttributeView.h"
#include "mesh/MeshStore.h"

#include "Range.h"
#include "gpu/PBRMaterial.h"

#include <span>

#include "state/Entity.h"

namespace mtl {
struct BufferContext;
} // namespace mtl
struct Mesh;

std::span<const PBRMaterial> GetMaterials(const state::Scene &);
// Returns store corners for triangle meshes or the triangulated index-arena range for n-gons.
TriangleVertexView GetFaceIndices(const state::Scene &, const Mesh &);
mtl::BufferContext &GetBufferContext(state::Scene &);
// Updates the posed bounds of the meshlet blocks holding the ascending meshlet addresses, at the owner's meshlet revision.
void UpdatePosedMeshletBlocks(state::Scene &, const MeshStore::Record &owner, std::span<const uint32_t> addresses);

void FreeInstanceRange(state::Scene &, Range);
