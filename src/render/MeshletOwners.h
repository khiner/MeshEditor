#pragma once
#include "Range.h"
#include "gpu/MeshletOwnersPushConstants.h"
#include "mesh/MeshStore.h"
#include <cstdint>
#include <span>
#include <vector>

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Scene;
} // namespace state

// Prepares publishing the finest cluster of every element the clusters' payloads name in the given topology, as runs of the owner's meshlet handles.
// Blocks names the record's local element blocks those elements occupy, which the owner range covers.
// Appends one job per cluster run, whose owner values RecordMeshletOwners writes.
void PrepareMeshletOwners(state::Scene &, MeshStore::Record &owner, uint32_t topology, std::span<const Range> clusters, std::span<const uint32_t> blocks, std::vector<MeshletOwnersJob> &);
// Records the owner writes of every job in one dispatch, with one threadgroup per cluster.
void RecordMeshletOwners(state::Scene &, mtl::ComputeChain &, std::span<const MeshletOwnersJob>);

// Clears the owner entries that name one of the clusters among the elements their payloads name, and releases the owner range once no block holds an owner.
// Retired lists the handles of erased elements the clusters name, whose blocks may have left the record's table.
// The clusters' records and payloads stay allocated until retirement returns.
// Retire before publishing replacement owners.
void RetireMeshletOwners(state::Scene &, MeshStore::Record &owner, std::span<const uint32_t> clusters, std::span<const uint32_t> retired);
