#pragma once
#include "gpu/MeshletOwnersPushConstants.h"
#include "mesh/MeshStore.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

// Prepares publishing the finest cluster of every element the clusters' payloads name in the given topology, as runs of the owner's meshlet handles.
// Blocks names the record's local element blocks those elements occupy, which the owner range covers.
// Appends one job per cluster run, whose owner values RecordMeshletOwners writes.
void PrepareMeshletOwners(state::Registry &, MeshStore::Record &owner, uint32_t topology, std::span<const Range> clusters, std::span<const uint32_t> blocks, std::vector<MeshletOwnersJob> &);
// Records the owner writes of every job in one dispatch, with one threadgroup per cluster.
void RecordMeshletOwners(state::Registry &, mtl::ComputeChain &, std::span<const MeshletOwnersJob>);

// Clear element-owner entries for these clusters and release empty owner ranges.
// Retired lists the handles of erased elements the clusters name, whose blocks may have left the record's table.
// The clusters' records and payloads stay allocated until retirement returns.
// Retire before publishing replacement owners.
void RetireMeshletOwners(state::Registry &, MeshStore::Record &owner, std::span<const uint32_t> clusters, std::span<const uint32_t> retired);
