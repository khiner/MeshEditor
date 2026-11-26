#include "mesh/Flatten.h"
#include "SortUnique.h"
#include "mesh/AttributeFields.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/SelectedChains.h"
#include "metal/Dispatch.h"

#include <unordered_set>

namespace {
// Faces holds indices into the selection's faces.
struct Group {
    std::vector<uint32_t> Vertices, Faces;
};
// Packs the groups of the selected faces and wires into the plan.
FlattenPlan Pack(const GeometrySelection &selection, const Gathered<FaceCorner> &corners, std::vector<Group> groups) {
    FlattenPlan plan;
    for (auto &group : groups) {
        SortUnique(group.Vertices);
        plan.Vertices.insert(plan.Vertices.end(), group.Vertices.begin(), group.Vertices.end());
    }
    SortUnique(plan.Vertices);
    std::unordered_map<uint32_t, uint32_t> ranks;
    for (uint32_t i = 0u; i < plan.Vertices.size(); ++i) ranks.emplace(plan.Vertices[i], i);
    std::vector<uint32_t> next_depth(plan.Vertices.size());
    std::vector<std::vector<uint32_t>> levels;
    for (const auto &group : groups) {
        uint32_t depth = 0u;
        for (const auto v : group.Vertices) depth = std::max(depth, next_depth[ranks.at(v)]);
        if (levels.size() <= depth) levels.resize(depth + 1u);
        levels[depth].push_back(uint32_t(plan.Words.size()));
        plan.Words.push_back(uint32_t(group.Vertices.size()));
        plan.Words.push_back(uint32_t(group.Faces.size()));
        for (const auto v : group.Vertices) {
            const auto rank = ranks.at(v);
            plan.Words.push_back(rank);
            next_depth[rank] = depth + 1u;
        }
        const auto offsets = uint32_t(plan.Words.size());
        plan.Words.resize(plan.Words.size() + group.Faces.size());
        for (uint32_t i = 0u; i < group.Faces.size(); ++i) {
            plan.Words[offsets + i] = uint32_t(plan.Words.size());
            const auto face_corners = corners.Of(group.Faces[i]);
            plan.Words.push_back(selection.Faces[group.Faces[i]]);
            plan.Words.push_back(uint32_t(face_corners.size()));
            for (const auto &corner : face_corners) plan.Words.push_back(ranks.at(corner.Vertex));
        }
    }
    for (const auto &level : levels) {
        plan.Batches.push_back({uint32_t(plan.Groups.size()), uint32_t(level.size())});
        plan.Groups.insert(plan.Groups.end(), level.begin(), level.end());
    }
    return plan;
}
} // namespace

void RecordPlanFlatten(const MeshStore &meshes, const MeshPipelines &pipelines, mtl::ComputeChain &chain, uint32_t id, const GeometrySelection &selection, std::function<void(FlattenPlan)> then) {
    RecordFaceCorners(meshes, pipelines, chain, id, selection.Faces, [&meshes, &pipelines, &chain, id, selection, then = std::move(then)](Gathered<FaceCorner> corners) mutable {
        RecordComponentLabels(meshes, pipelines, chain, id, Element::Face, selection.Faces, FieldAdjacency::Radial, false, [&meshes, &pipelines, &chain, id, selection = std::move(selection), corners = std::move(corners), then = std::move(then)](std::vector<uint32_t> face_labels) mutable {
            // Each group by the label of its component, in the order the components first appear.
            auto groups = std::make_shared<std::vector<Group>>();
            std::unordered_map<uint32_t, uint32_t> components;
            const auto group_of = [&](uint32_t label) -> Group & {
                const auto [entry, inserted] = components.try_emplace(label, uint32_t(groups->size()));
                if (inserted) groups->emplace_back();
                return (*groups)[entry->second];
            };
            std::unordered_set<uint32_t> covered_edges;
            for (uint32_t i = 0u; i < selection.Faces.size(); ++i) {
                auto &group = group_of(face_labels[i]);
                group.Faces.push_back(i);
                for (const auto &corner : corners.Of(i)) {
                    group.Vertices.push_back(corner.Vertex);
                    covered_edges.insert(corner.Edge);
                }
            }
            std::vector<uint32_t> wires;
            for (const auto e : selection.Edges) {
                if (!covered_edges.contains(e)) wires.push_back(e);
            }
            auto facts = std::make_shared<std::vector<EdgeFacts>>();
            RecordSelectedEdgeFacts(meshes, pipelines, chain, id, wires, {}, [facts](std::vector<EdgeFacts> read) { *facts = std::move(read); });
            RecordComponentLabels(meshes, pipelines, chain, id, Element::Edge, wires, FieldAdjacency::Domain, false, [facts, groups, selection = std::move(selection), corners = std::move(corners), then = std::move(then)](std::vector<uint32_t> edge_labels) {
                std::unordered_map<uint32_t, uint32_t> components;
                for (uint32_t i = 0u; i < edge_labels.size(); ++i) {
                    const auto [entry, inserted] = components.try_emplace(edge_labels[i], uint32_t(groups->size()));
                    if (inserted) groups->emplace_back();
                    auto &vertices = (*groups)[entry->second].Vertices;
                    vertices.insert(vertices.end(), {(*facts)[i].From, (*facts)[i].To});
                }
                then(Pack(selection, corners, std::move(*groups)));
            });
        });
    });
}
