#pragma once
#include "mesh/MeshStore.h"

template<typename T> struct BufferArena;
namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Registry;
} // namespace state

struct MeshletBoundsRefitJob {
    const MeshStore::Record *Owner;
    const BufferArena<uint32_t> *Storage;
    ElementWork Meshlets; // The meshlets to refit, at their arena addresses
};

// Records the refit of only the finest meshlets affected by canonical vertex-position writes.
// The owners' spatial trees refit on the host once the chain submits.
void RefitCanonicalMeshletBounds(state::Registry &, mtl::ComputeChain &, std::span<const MeshletBoundsRefitJob>);
// Records the refit of each entity's moved finest clusters' traversal leaves and marks their coarse ancestors stale.
void StageDirtyPositionMeshlets(state::Registry &, mtl::ComputeChain &, std::span<const state::Entity>);
