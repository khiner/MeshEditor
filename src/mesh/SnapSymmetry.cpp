#include "mesh/SnapSymmetry.h"
#include "Profile.h"
#include "mesh/Mesh.h"
#include "mesh/Neighborhood.h"
#include <unordered_set>

void RecordPlanSymmetrySnap(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, const GeometrySelection &selection, uint32_t axis, float threshold, bool center, std::function<void(std::vector<SymmetrySnapVertex>)> then) {
    if (selection.Vertices.empty()) return then({});
    auto mirrored = VertexPositions(meshes, id, selection.Vertices);
    for (auto &position : mirrored) position[axis] = -position[axis];
    RecordNearestVertices(meshes, pipelines, chain, id, mirrored, threshold, [vertices = selection.Vertices, center, then = std::move(then)](std::vector<uint32_t> partners) {
        const profile::CpuScope scope{"PlanSymmetrySnap"};
        std::unordered_set<uint32_t> used;
        std::vector<SymmetrySnapVertex> output;
        for (uint32_t i = 0u; i < vertices.size(); ++i) {
            const auto v = vertices[i], partner = partners[i];
            if (used.contains(v) || partner == InvalidOffset || used.contains(partner) || (partner == v && !center)) continue;
            // Ambiguous neighborhoods use the first canonical selected pair, so each destination is written once, including coincident or overlapping pairs.
            used.insert(v);
            used.insert(partner);
            output.push_back({v, partner});
            if (partner != v) output.push_back({partner, v});
        }
        std::ranges::sort(output, {}, &SymmetrySnapVertex::Vertex);
        profile::RecordCounter("SymmetrySnapPairs", output.size());
        then(std::move(output));
    });
}
