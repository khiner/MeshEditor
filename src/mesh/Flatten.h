#pragma once
#include "mesh/GeometrySelection.h"

#include "Range.h"

#include <functional>

struct Mesh;
struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}

// A group record in Words holds its vertex count, face count, vertex ranks and face record offsets into Words.
// A face record holds its canonical face, corner count and vertex ranks.
// Batches hold independent groups in dependency order.
struct FlattenPlan {
    std::vector<uint32_t> Vertices, Words, Groups;
    std::vector<Range> Batches;
};
// The selected faces joined across their edges and the other selected edges joined across their vertices, each group flattened to one plane.
// Records the reads in the record's lanes of the chain, each once the chain completed the reads it follows, and calls `then` with the plan once they completed.
void RecordPlanFlatten(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, std::function<void(FlattenPlan)> then);
