#pragma once

#include "Range.h"
#include <vector>

struct Mesh;
struct MeshStore;

// Group records: vertex count, face count, vertex ranks, face-record offsets.
// Face records: canonical face, corner count, vertex ranks.
// Offsets address Words. Batches contain independent groups in dependency order.
struct FlattenPlan {
    std::vector<uint32_t> Vertices, Words, Groups;
    std::vector<Range> Batches;
};
FlattenPlan PlanFlatten(const MeshStore &, const Mesh &);
