#pragma once

#include "gpu/ElementWork.h"
#include "gpu/MeshRecord.h"
#include "gpu/SlotOffset.h"
#include "state/Entity.h"
#include <span>

struct GpuBuffers;
struct MeshBuffers;
struct MeshStore;
namespace mtl { struct ComputeChain; }

// A mesh or procedural mesh uses the same GPU builder. Primitive membership,
// positions, topology and attributes are read from canonical GPU arenas.
struct MeshletBuildSource {
    MeshBuffers *Destination;
    MeshRecord Mesh;
    uint32_t StoreId{InvalidOffset};
    SlotOffset AuxIndices;
    uint32_t Topology, ElementCount;
    ElementWork Elements{}; // Optional finished canonical triangle, edge or point membership.
    // A local fragment owns only its new meshlets and payloads. Existing
    // primitive/group records are borrowed until the caller adopts the fragment.
    const MeshBuffers *Owner{};
    uint32_t Primitive{InvalidOffset}, Group{InvalidOffset};
};

// Membership and element work live in the chain's scratch, which the build reserves once.
// The build submits the chain for its meshlet counts.
// A build without an owner also submits for its gathered counts and for its publication, which publishes a canonical source's element owners.
void BuildGpuMeshlets(state::Scene &, mtl::ComputeChain &, std::span<MeshletBuildSource>);
// The chain scratch words BuildGpuMeshlets allocates for `sources`.
// The count excludes an owned source's elements, so a caller can take it before seeding them.
uint64_t MeshletBuildScratchWords(const MeshStore &, std::span<const MeshletBuildSource>);

MeshRecord BuildMeshRecord(const GpuBuffers &, const MeshBuffers &, const MeshStore &, uint32_t store_id, bool face_topology, bool line_topology);

// Create one empty primitive/LOD leaf only when a local build first uses its
// source material. Existing routes and every untouched primitive stay in place.
uint32_t EnsureMeshletPrimitive(state::Scene &, MeshBuffers &, uint32_t source_primitive);
