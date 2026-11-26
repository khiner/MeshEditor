#pragma once
#include "mesh/GeometrySelection.h"

#include <functional>

struct MeshStore;
struct SymmetrySnapVertex {
    uint32_t Vertex, Partner;
};
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}
// Records the reads of the disjoint nearest mirror pairs within the threshold in the record's lane of the chain, which gather with the other pending lanes' reads, and calls `then` with them, in canonical handle order and including unselected counterparts, after the submit that completes them.
void RecordPlanSymmetrySnap(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, uint32_t axis, float threshold, bool center, std::function<void(std::vector<SymmetrySnapVertex>)> then);
