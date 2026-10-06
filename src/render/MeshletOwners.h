#pragma once
#include "Range.h"
#include "mesh/MeshStore.h"
#include <cstdint>
#include <span>

namespace mtl {
struct ComputeChain;
}
namespace state {
struct Scene;
}

// Publishes the finest cluster of every element the clusters' payloads name in the given topology.
// Blocks names the canonical element blocks those elements occupy, which the host attaches owner payloads to.
// The chain writes the owner values.
void PublishMeshletOwners(state::Scene &, mtl::ComputeChain &, MeshStore::Record &owner, uint32_t topology, std::span<const Range> clusters, std::span<const uint32_t> blocks);

// Clears the owner entries that name one of the clusters among the elements their payloads name, then releases the blocks left without an owner.
// The clusters' records and payloads stay allocated until retirement returns.
// Retire before publishing replacement owners.
void RetireMeshletOwners(state::Scene &, MeshStore::Record &owner, std::span<const uint32_t> clusters);
