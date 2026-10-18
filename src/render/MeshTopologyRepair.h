#pragma once
#include "state/Entity.h"
#include <span>
#include <vector>
struct MeshTopologyEdit;
struct FaceTriangles;
struct MeshletBuildSource;
namespace mtl {
struct ComputeChain;
} // namespace mtl

// Publish each entity's in-place topology edit through finest payload, simplification dependencies and traversal bounds before retiring its source identities.
// The affected coarse groups turn stale until the next coarse repair.
// The repairs share one fragment build on the edits' chain, which submits once where the host reads counts and records, and whose owner submits the rest.
// Batch fresh builds with repair builds to publish their element owners together.
void RepairTopologyRender(state::Registry &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, const MeshTopologyEdit *>>, std::span<MeshletBuildSource> fresh = {});
// Batch finest-render repairs for changed shading or tessellation in one build and submission.
void RepairFaceRender(state::Registry &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, FaceTriangles>>);
// A point or line record's affected vertices or edges, at their arena addresses before the edit erased any of them.
struct ElementMeshletRepair {
    uint32_t StoreId;
    uint32_t Topology;
    std::vector<uint32_t> Elements; // Handles of the record's affected elements, live or retired
};
// Moves the affected vertices of each point record, or the affected edges of each line record, between its owner's clusters, which draw every live element.
// Retired elements leave their clusters, and live elements without one build as fragments the owners adopt, in one build recorded on the chain.
void RepairElementMeshlets(state::Registry &, mtl::ComputeChain &, std::span<const ElementMeshletRepair>);
