#pragma once
#include "mesh/MeshStore.h"

namespace mtl {
struct ComputeChain;
} // namespace mtl

// A triangle edit's canonical membership, as finished work in one storage arena.
struct MeshletPatchInput {
    ElementWork Changed{}, Removed{}, ReplacedCorners{};
    // The emitted triangle run and the source triangle each emitted triangle inherits its render owner from.
    Range Added{};
    std::span<const uint32_t> Sources{};
    // Unchanged uniform render keys keep a surviving triangle's cluster unless it renders a corner the edit retired.
    bool StableKeys{};
};

// The elements rebuilt under one existing primitive, group and traversal leaf.
struct MeshletPatchPartition {
    uint32_t Group{InvalidOffset}, Primitive{InvalidOffset}, Leaf{InvalidOffset};
    std::vector<uint32_t> Elements;
};

// The finest clusters a triangle edit retires, the LOD groups they belong to, and the partitions that keep triangles.
// Partitions come in ascending group then primitive order.
struct MeshletPatch {
    std::vector<uint32_t> Clusters, Groups;
    std::vector<MeshletPatchPartition> Partitions;
};

// Plans a triangle edit's render repair on the host from the owner's triangle owners and cluster payloads.
// Source owners and cluster/group records must remain live throughout.
// Faces with no source triangle use their material route, creating it when needed.
MeshletPatch PlanMeshletPatch(state::Registry &, MeshStore::Record &, const BufferArena<uint32_t> &storage, const MeshletPatchInput &);

// One fragment's clusters and the primitive, group and traversal leaf they join.
struct MeshletPatchAdoptJob {
    uint32_t First{}, Count{};
    uint32_t ElementCount{}; // The elements the fragment's clusters draw
    uint32_t Group{InvalidOffset}, Primitive{InvalidOffset};
    uint32_t Leaf{InvalidOffset};
};

// Adopt locally built, primitive-bound finest meshlets into one canonical owner.
// The host publishes their membership, traversal leaves and element owner blocks, and the chain writes their owner values.
// Return canonical element blocks in Blocks and empty the fragments.
// Returns the adopted clusters in ascending order.
std::vector<uint32_t> AdoptMeshletFragments(state::Registry &, mtl::ComputeChain &, MeshStore::Record &, uint32_t topology, std::span<MeshStore::Record> fragments, std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks);
