#pragma once
#include "mesh/GeometrySelection.h"

#include "Range.h"
#include "gpu/VertexPositionEditPushConstants.h"
#include <functional>
#include <unordered_map>
#include <vector>

struct Mesh;

using EdgeGraph = std::unordered_map<uint32_t, std::vector<uint32_t>>;
// Visit maximal paths in canonical order: endpoints/junctions first, then cycles.
void VisitEdgeChains(EdgeGraph &, const std::function<void(const std::vector<uint32_t> &, bool closed)> &);

// Selected topology only. Positions and spline coefficients remain on the GPU.
struct EdgeChainPlan {
    std::vector<uint32_t> Inputs, Outputs, Phases;
    std::vector<EdgeChain> Chains;
    std::vector<Range> Batches; // Ordered nonconflicting paths
};
EdgeChainPlan PlanSelectedEdgeChains(const Mesh &, const GeometrySelection &, bool relax);

// Outputs are unique canonical handles; Inputs index Outputs. Phases store knot/point
// counts, knot indices, point indices, and each point's surface membership.
EdgeChainPlan PlanCurveBetweenSelected(const Mesh &, const GeometrySelection &, const GeometrySelection &excluded, bool extend);

// Selected surface boundaries and isolated wire chains, using shared position ranks.
EdgeChainPlan PlanCircularize(const Mesh &, const GeometrySelection &, const GeometrySelection &excluded = {});
