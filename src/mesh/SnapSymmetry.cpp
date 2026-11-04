#include "mesh/SnapSymmetry.h"
#include "Profile.h"
#include "mesh/Mesh.h"
#include "mesh/Neighborhood.h"
#include <unordered_set>

std::vector<SymmetrySnapVertex> PlanSymmetrySnap(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const GeometrySelection &selection, uint32_t axis, float threshold, bool center) {
    const profile::CpuScope scope{"PlanSymmetrySnap"};
    if (selection.Vertices.empty()) return {};
    auto mirrored = VertexPositions(meshes, pipelines, mesh.GetStoreId(), selection.Vertices);
    for (auto &position : mirrored) position[axis] = -position[axis];
    const auto partners = NearestVertices(meshes, pipelines, mesh.GetStoreId(), mirrored, threshold);
    std::unordered_set<uint32_t> used;
    std::vector<SymmetrySnapVertex> output;
    for (uint32_t i = 0u; i < selection.Vertices.size(); ++i) {
        const auto v = selection.Vertices[i], partner = partners[i];
        if (used.contains(v) || partner == InvalidOffset || used.contains(partner) || (partner == v && !center)) continue;
        // Ambiguous neighborhoods use the first canonical selected pair, so each destination is written once, including coincident or overlapping pairs.
        used.insert(v);
        used.insert(partner);
        output.push_back({v, partner});
        if (partner != v) output.push_back({partner, v});
    }
    std::ranges::sort(output, {}, &SymmetrySnapVertex::Vertex);
    profile::RecordCounter("SymmetrySnapPairs", output.size());
    return output;
}
