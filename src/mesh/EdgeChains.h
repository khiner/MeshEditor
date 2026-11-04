#pragma once
#include "mesh/GeometrySelection.h"

#include "Range.h"
#include "gpu/VertexPositionEditPushConstants.h"

struct Mesh;
struct MeshStore;
struct MeshPipelines;

// Outputs are unique canonical handles and Inputs index Outputs.
struct EdgeChainPlan {
    std::vector<uint32_t> Inputs, Outputs, Phases;
    std::vector<EdgeChain> Chains;
    std::vector<Range> Batches; // Ordered nonconflicting paths
};
// The selected edges' chains of three or more vertices, found on the GPU.
EdgeChainPlan PlanSelectedEdgeChains(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &, bool relax);

// Phases hold the knot and point counts, the knot indices, the point indices and each point's surface membership.
EdgeChainPlan PlanCurveBetweenSelected(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &, const GeometrySelection &excluded, bool extend);

// Selected surface boundaries and isolated wire chains.
EdgeChainPlan PlanCircularize(const MeshStore &, const MeshPipelines &, const Mesh &, const GeometrySelection &, const GeometrySelection &excluded);
