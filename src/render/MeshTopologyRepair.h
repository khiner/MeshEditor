#pragma once
#include "state/Entity.h"
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
struct MeshTopologyEdit;
struct FaceTriangles;
struct MeshletBuildSource;
namespace mtl {
struct ComputeChain;
}

// Publish each entity's in-place topology edit through finest payload, simplification
// dependencies and traversal bounds before retiring its source identities.
// The affected coarse groups turn stale until the next coarse repair.
// The repairs share one fragment build on the edits' chain, which submits once where the host reads counts and records, and whose owner submits the rest.
// Fresh builds of other meshes ride the repairs' meshlet build, which publishes their element owners.
void RepairTopologyRender(state::Scene &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, const MeshTopologyEdit *>>, std::span<MeshletBuildSource> fresh = {});
// Repairs the finest render of each entity's changed triangles after their shading or tessellation changes, with one fragment build and one submit for all of them.
void RepairFaceRender(state::Scene &, mtl::ComputeChain &, std::span<const std::pair<state::Entity, FaceTriangles>>);
// A point or line record's affected vertices or edges.
struct ElementMeshletRepair {
    uint32_t StoreId;
    uint32_t Topology;
    std::vector<uint32_t> Elements;
};
// Moves the affected vertices of each point record, or the affected edges of each line record, between its owner's clusters, which draw every live element.
// Retired elements leave their clusters, and live elements without one build as fragments the owners adopt, in one build recorded on the chain.
void RepairElementMeshlets(state::Scene &, mtl::ComputeChain &, std::span<const ElementMeshletRepair>);
