#pragma once
#include "mesh/MeshStore.h"
#include "Range.h"
#include "gpu/ElementWork.h"
#include "metal/BufferArena.h"
#include "state/Entity.h"
#include <span>
#include <vector>

namespace mtl { struct ComputeChain; }

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
MeshletPatch PlanMeshletPatch(state::Scene &, const MeshStore::Record &, const BufferArena<uint32_t> &storage, const MeshletPatchInput &);

// One fragment's clusters and the primitive, group and traversal leaf they join.
struct MeshletPatchAdoptJob {
    uint32_t First{}, Count{};
    uint32_t Group{InvalidOffset}, Primitive{InvalidOffset};
    uint32_t Leaf{InvalidOffset};
};

// Adopt locally built, primitive-bound finest meshlets into one canonical owner.
// The host publishes their membership, traversal leaves and element owner blocks, and the chain writes their owner values.
// Blocks names the canonical element blocks the fragments hold, and the fragments become empty.
// Returns the adopted clusters in ascending order.
std::vector<uint32_t> AdoptMeshletFragments(state::Scene &, mtl::ComputeChain &, MeshStore::Record &, std::span<MeshStore::Record> fragments,
                                            std::span<const MeshletPatchAdoptJob> jobs, std::span<const uint32_t> blocks);
