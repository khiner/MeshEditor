#pragma once
#include "mesh/MeshStore.h"

#include "gpu/PBRMaterial.h"

std::span<const PBRMaterial> GetMaterials(const state::Registry &);
// Updates the posed bounds of the meshlet blocks holding the ascending meshlet addresses, at the owner's meshlet revision.
void UpdatePosedMeshletBlocks(state::Registry &, const MeshStore::Record &owner, std::span<const uint32_t> addresses);

void FreeInstanceRange(state::Registry &, Range);
