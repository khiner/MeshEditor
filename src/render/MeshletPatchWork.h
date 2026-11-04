#pragma once
#include "mesh/MeshStore.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl

struct MeshletPatchInput {
    // The ascending finest clusters the edit retires.
    std::span<const uint32_t> Clusters{};
    ElementWork Removed{};
    // The emitted triangles as ascending handle runs, and per emitted triangle in run order the cluster it inherits its render owner from.
    // A source of InvalidOffset takes the face's material route.
    std::span<const Range> Added{};
    std::span<const uint32_t> Sources{};
};

// The ascending finest clusters owning a triangle edit's changed triangles that keep no cluster, from finished work in one storage arena.
std::vector<uint32_t> ChangedClusters(state::Registry &, const MeshStore::Record &, const BufferArena<uint32_t> &storage, ElementWork changed, ElementWork removed, ElementWork replaced_corners, bool stable_keys);

struct MeshletPatchPartition {
    uint32_t Group{InvalidOffset}, Primitive{InvalidOffset}, Leaf{InvalidOffset};
    std::vector<uint32_t> Elements;
};

// The LOD groups of the finest clusters a triangle edit retires, and the partitions that keep triangles.
// Partitions come in ascending group then primitive order.
struct MeshletPatch {
    std::vector<uint32_t> Groups;
    std::vector<MeshletPatchPartition> Partitions;
};

// Plans a triangle edit's render repair on the host from the owner's triangle owners and cluster payloads.
// Source owners and cluster/group records must remain live throughout.
MeshletPatch PlanMeshletPatch(state::Registry &, MeshStore::Record &, const BufferArena<uint32_t> &storage, const MeshletPatchInput &);

struct MeshletPatchAdoptJob {
    uint32_t First{}, Count{};
    uint32_t ElementCount{};
    uint32_t Group{InvalidOffset}, Primitive{InvalidOffset};
    uint32_t Leaf{InvalidOffset};
};

// Adopts locally built, primitive-bound finest meshlets into one canonical owner, taking its canonical element blocks from `blocks`, and empties the fragments.
// Returns the adopted clusters in ascending order.
std::vector<uint32_t> AdoptMeshletFragments(state::Registry &, mtl::ComputeChain &, MeshStore::Record &, uint32_t topology, std::span<MeshStore::Record> fragments, std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks);
