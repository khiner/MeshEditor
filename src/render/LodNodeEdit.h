#pragma once
#include <cstdint>
#include <span>

struct MeshBuffers;
namespace mtl { struct ComputeChain; }
namespace state { struct Scene; }

// Clusters [First, First + Count) that join the owner under one primitive.
// A finest run also joins the primitive's pinned finest node.
struct LodClusterRun {
    uint32_t First{}, Count{}, Primitive{};
    bool Finest{};
};

// Moves the removed and added clusters' traversal-leaf and pinned-finest memberships, then refits every node above them.
// Added clusters already belong to the owner and name their intended leaf, and their records can still be pending on the chain.
// Removed clusters and their reverse links stay allocated until the edit returns.
// Touched clusters keep their memberships and refit their leaves and ancestors.
// The host writes memberships, member counts and primitive totals, and the chain records the bounds refit, deepest nodes first.
// A leaf past twice the build's leaf span splits into even runs of at most one span, and a node past twice the build's width splits the same way.
// An emptied leaf, and a node its removal leaves without children, leave their parents and release their ids, while a primitive's root stays.
// Each change moves the parent's children to one new contiguous run, and a root that splits adds a level below itself, so each tree stays as balanced as the build makes it.
// Coarse geometry and group errors stay as they are.
void EditLodNodes(state::Scene &, mtl::ComputeChain &, MeshBuffers &, std::span<const uint32_t> removed, std::span<const LodClusterRun> added,
                  std::span<const uint32_t> touched);
