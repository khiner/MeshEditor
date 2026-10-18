#pragma once
#include "mesh/MeshStore.h"

#include "gpu/PBRMaterial.h"

namespace mtl {
struct BufferContext;
} // namespace mtl
struct Mesh;

std::span<const PBRMaterial> GetMaterials(const state::Registry &);
// Returns store corners for triangle meshes or the triangulated index-arena range for n-gons.
TriangleVertexView GetFaceIndices(const state::Registry &, const Mesh &);
mtl::BufferContext &GetBufferContext(state::Registry &);
// Updates the posed bounds of the meshlet blocks holding the ascending meshlet addresses, at the owner's meshlet revision.
void UpdatePosedMeshletBlocks(state::Registry &, const MeshStore::Record &owner, std::span<const uint32_t> addresses);

void FreeInstanceRange(state::Registry &, Range);
