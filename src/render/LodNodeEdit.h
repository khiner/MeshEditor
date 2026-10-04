#pragma once
#include "Range.h"
#include "gpu/LodNodeRefitPushConstants.h"
#include <cstdint>
#include <span>
#include <vector>

struct MeshBuffers;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

// Clusters [First, First + Count) that join the owner under one primitive.
// A finest run also joins the primitive's pinned finest node.
struct LodClusterRun {
    uint32_t First{}, Count{}, Primitive{};
    bool Finest{};
};

// One owner's node bounds refit: the nodes at each depth, as job ranges of the chain's scratch.
struct LodNodeRefit {
    LodNodeRefitPushConstants Pc{};
    std::vector<std::pair<uint32_t, Range>> Depths{};
};

// Moves the removed and added clusters' traversal-leaf and pinned-finest memberships, then plans the refit of every node above them.
// Added clusters already belong to the owner and name their intended leaf, and their records can still be pending on the chain.
// Removed clusters and their reverse links stay allocated until the edit returns.
// Touched clusters keep their memberships and refit their leaves and ancestors.
// The host writes memberships, member counts and primitive totals, and RecordLodNodeRefits records the returned bounds refit.
// A leaf past twice the build's leaf span splits into even runs of at most one span, and a node past twice the build's width splits the same way.
// An emptied leaf, and a node its removal leaves without children, leave their parents and release their ids, while a primitive's root stays.
// Each change moves the parent's children to one new contiguous run, and a root that splits adds a level below itself, so each tree stays as balanced as the build makes it.
// Coarse geometry and group errors stay as they are.
[[nodiscard]] LodNodeRefit EditLodNodes(state::Scene &, mtl::ComputeChain &, MeshBuffers &, std::span<const uint32_t> removed, std::span<const LodClusterRun> added,
                                       std::span<const uint32_t> touched);
// Records the refits depth by depth, deepest first, with every owner's nodes at a depth in one concurrent group.
// A node at a depth reads only its children's bounds, which the deeper groups write first.
void RecordLodNodeRefits(state::Scene &, mtl::ComputeChain &, std::span<const LodNodeRefit>);
