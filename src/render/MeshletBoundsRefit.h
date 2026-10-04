#pragma once
#include "mesh/MeshStore.h"

#include "gpu/ElementWork.h"
#include "state/Entity.h"
#include <cstdint>
#include <span>

template<typename T> struct BufferArena;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

struct MeshletBoundsRefitJob {
    const MeshStore::Record *Owner;
    const BufferArena<uint32_t> *Storage;
    ElementWork Meshlets;
};

// Records the refit of only the finest meshlets affected by canonical vertex-position writes.
// The owners' spatial trees refit on the host once the chain submits.
void RefitCanonicalMeshletBounds(state::Scene &,mtl::ComputeChain &,std::span<const MeshletBoundsRefitJob>);
// Records the refit of each entity's moved finest clusters' traversal leaves and marks their coarse ancestors stale.
void StageDirtyPositionMeshlets(state::Scene &,mtl::ComputeChain &,std::span<const state::Entity>);
