#pragma once
#include "Range.h"
#include <cstdint>
#include <span>

struct MeshBuffers;
struct MeshStore;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

// The first element handle of the store record's domain in the render topology, which its element owner blocks start from.
uint32_t ElementDomainFirst(const MeshStore &, uint32_t store_id, uint32_t topology);

// Publishes the finest cluster of every element the clusters' payloads name, in the owner's render topology.
// Blocks names the canonical element blocks those elements occupy, which the host attaches owner payloads to.
// The chain writes the owner values.
void PublishMeshletOwners(state::Scene &, mtl::ComputeChain &, MeshBuffers &owner, std::span<const Range> clusters, std::span<const uint32_t> blocks);

// Clears the owner entries that name one of the clusters among the elements their payloads name, then releases the blocks left without an owner.
// The clusters' records and payloads stay allocated until retirement returns.
// Retire before publishing replacement owners.
void RetireMeshletOwners(state::Scene &, MeshBuffers &owner, std::span<const uint32_t> clusters);
