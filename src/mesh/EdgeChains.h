#pragma once
#include "mesh/GeometrySelection.h"

#include "Range.h"
#include "gpu/VertexPositionEditPushConstants.h"

#include <functional>

struct Mesh;
class LocalMesh;
struct MeshStore;
struct MeshPipelines;
namespace mtl {
struct ComputeChain;
}

// Outputs are unique canonical handles and Inputs index Outputs.
struct EdgeChainPlan {
    std::vector<uint32_t> Inputs, Outputs, Phases;
    std::vector<EdgeChain> Chains;
    std::vector<Range> Batches; // Ordered nonconflicting paths
};
// Records the reads of the selected edges' chains of three or more vertices on the chain, which gather with the other pending lanes' reads, and calls `then` with the plan after the submits that complete them.
void RecordPlanSelectedEdgeChains(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, bool relax, std::function<void(EdgeChainPlan)> then);

// Phases hold the knot and point counts, the knot indices, the point indices and each point's surface membership.
// It reads the local mesh around every vertex of the record.
EdgeChainPlan PlanCurveBetweenSelected(const LocalMesh &, const GeometrySelection &, const GeometrySelection &excluded, bool extend);

// Selected surface boundaries and isolated wire chains.
// Records its reads on the chain, which gather with the other pending lanes' reads, and calls `then` with the plan after the submits that complete them.
void RecordPlanCircularize(const MeshStore &, const MeshPipelines &, mtl::ComputeChain &, uint32_t id, const GeometrySelection &, const GeometrySelection &excluded, std::function<void(EdgeChainPlan)> then);
