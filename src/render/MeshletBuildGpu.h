#pragma once

#include "mesh/MeshStore.h"

struct GpuBuffers;
namespace mtl {
struct ComputeChain;
} // namespace mtl

// A mesh or procedural mesh uses the same GPU builder.
// Primitive membership, positions, topology and attributes are read from canonical GPU arenas.
// The destination names the store record, which an extras record reads as bone or joint faces.
// A canonical destination may have one source per topology; all publish together.
struct MeshletBuildSource {
    MeshStore::Record *Destination;
    uint32_t Topology, ElementCount; // Exact for supplied Elements, otherwise the canonical domain count before GPU filtering.
    ElementWork Elements{}; // Optional finished triangle, wire edge or isolated point membership.
    // A local fragment owns only its new meshlets, which join the owner's meshlet set, and their payloads.
    // Existing primitive/group records are borrowed until the caller adopts the fragment.
    MeshStore::Record *Owner{};
    uint32_t Primitive{InvalidOffset}, Group{InvalidOffset};
};

// Membership and element work live in the chain's scratch, which each chunk of sources reserves once.
// Each chunk submits the chain for its meshlet counts, and a chunk stays under the scratch budget unless one destination exceeds it alone.
// A build without an owner also submits for its gathered counts and for its publication, which publishes a canonical source's element owners.
void BuildGpuMeshlets(state::Registry &, mtl::ComputeChain &, std::span<MeshletBuildSource>);
// The chain scratch words BuildGpuMeshlets allocates for `sources`.
// The count excludes an owned source's elements, so a caller can take it before seeding them.
uint64_t MeshletBuildScratchWords(const MeshStore &, std::span<const MeshletBuildSource>);

// The shared GPU bindings of a store record, including face and vertex attributes.
MeshRecord BuildMeshRecord(const GpuBuffers &, const MeshStore &, uint32_t store_id);
// Rewrites the record's GPU mesh record from the current bindless slots.
void RefreshMeshBinding(state::Registry &, uint32_t store_id);

// Create one empty primitive/LOD leaf only when a local build first uses its source material.
// Existing routes and every untouched primitive stay in place.
uint32_t EnsureMeshletPrimitive(state::Registry &, MeshStore::Record &, uint32_t source_primitive, uint32_t topology);
