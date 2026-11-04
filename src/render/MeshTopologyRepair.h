#pragma once
#include "mesh/PageFootprint.h"
#include "state/Entity.h"
#include <span>
#include <vector>
struct MeshTopologyEdit;
struct FaceTriangles;
struct MeshletBuildSource;
namespace mtl {
struct ComputeChain;
} // namespace mtl

// Publishes each entity's in-place topology edit through finest payload, simplification dependencies and traversal bounds, then retires its source identities.
// The repairs share one fragment build on the edits' chain, which submits once for the host to read counts and records, and the chain's owner submits the rest.
// Builds in `fresh` join the same build so their element owners publish together.
void RepairTopologyRender(state::Registry &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, const MeshTopologyEdit *>>, std::span<MeshletBuildSource> fresh = {});
// The triangle changes of successive in-place topology edits of one record, whose render records none of them repaired.
struct AccumulatedTriangleEdits {
    IndexMarks Clusters;
    uint32_t ClusterCount{};
    // Bit words by triangle handle, eight to a block, of the held triangles the edits removed and of the live triangles they emitted.
    std::vector<uint32_t> Removed, Emitted;
    // Per emitted triangle handle, the cluster it inherits its render owner from, or InvalidOffset for its face's material route.
    std::vector<uint32_t> Sources;
};
// Adds a published edit's triangle changes before it finishes, while its held triangles keep their render owners.
void AccumulateTriangleEdits(state::Registry &, AccumulatedTriangleEdits &, const MeshTopologyEdit &);
// Publishes the accumulated edits of the entity's record in one repair of the record the last edit leaves.
void RepairTopologyRender(state::Registry &, mtl::ComputeChain &, state::Entity, const AccumulatedTriangleEdits &);
// Batches finest-render repairs for changed shading or tessellation into one build and submission.
void RepairFaceRender(state::Registry &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, FaceTriangles>>);
// A point or line record's affected vertices or edges, at their arena addresses before the edit erased any of them.
struct ElementMeshletRepair {
    uint32_t StoreId;
    uint32_t Topology;
    std::vector<uint32_t> Added, Removed; // Ascending handles of unowned elements that start drawing and of owned elements that stop drawing
};
// Moves the affected vertices of each point record, or the affected edges of each line record, between its owner's clusters, which draw every live element.
void RepairElementMeshlets(state::Registry &, mtl::ComputeChain &, std::span<const ElementMeshletRepair>);
