#pragma once
#include "mesh/GeometrySelection.h"

#include "Range.h"

struct Mesh;
struct MeshStore;
struct MeshPipelines;

// A group record in Words holds its vertex count, face count, vertex ranks and face record offsets into Words.
// A face record holds its canonical face, corner count and vertex ranks.
// Batches hold independent groups in dependency order.
struct FlattenPlan {
    std::vector<uint32_t> Vertices, Words, Groups;
    std::vector<Range> Batches;
};
// The selected faces joined across their edges and the other selected edges joined across their vertices, each group flattened to one plane.
FlattenPlan PlanFlatten(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &);
