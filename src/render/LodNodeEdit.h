#pragma once
#include "gpu/LodNodeRefitPushConstants.h"
#include "mesh/MeshStore.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl
namespace state {
struct Scene;
} // namespace state

// Clusters [First, First + Count) that join the owner under one primitive.
// A finest run also joins the primitive's pinned finest node.
struct LodClusterRun {
    uint32_t First{}, Count{}, Primitive{};
    bool Finest{};
};

// One owner's node bounds refit: each node's depth and job, deepest first.
struct LodNodeRefit {
    std::vector<std::pair<uint32_t, LodNodeRefitJob>> Nodes{};
};

// Update removed and added cluster memberships and return ancestor refits for RecordLodNodeRefits.
// Added clusters must belong to the owner and name their intended leaves; GPU records may still be pending.
// Retain removed clusters and reverse links until this returns.
// Touched clusters preserve membership; coarse geometry and group errors remain unchanged.
[[nodiscard]] LodNodeRefit EditLodNodes(state::Scene &, MeshStore::Record &, std::span<const uint32_t> removed, std::span<const LodClusterRun> added, std::span<const uint32_t> touched);
// Records the refits depth by depth, deepest first, with every owner's nodes at a depth in one dispatch.
// A node at a depth reads only its children's bounds, which the deeper dispatches write first.
void RecordLodNodeRefits(state::Scene &, mtl::ComputeChain &, std::span<const LodNodeRefit>);
