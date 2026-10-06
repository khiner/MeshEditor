#include "mesh/Flatten.h"
#include "SortUnique.h"
#include "mesh/Mesh.h"
#include "mesh/MeshStore.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace {
struct Group {
    std::vector<uint32_t> Vertices, Faces;
};
} // namespace

FlattenPlan PlanFlatten(const MeshStore &meshes, const Mesh &mesh) {
    std::vector<Group> groups;
    std::unordered_set<uint32_t> covered_edges;
    // Face groups join across edges. The remaining selected edges join across vertices.
    for (const auto domain : {Element::Face, Element::Edge}) {
        std::vector<uint32_t> elements, parent;
        std::unordered_map<uint32_t, uint32_t> users;
        const auto root = [&](uint32_t i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        meshes.GetSelectedElements(mesh.GetStoreId(), domain).ForEach([&](uint32_t handle) {
            if (domain == Element::Edge && covered_edges.contains(handle)) return;
            const auto i = uint32_t(elements.size());
            elements.push_back(handle);
            parent.push_back(i);
            const auto join = [&](uint32_t key) {
                const auto [entry, inserted] = users.try_emplace(key, i);
                if (!inserted) {
                    const auto a = root(i), b = root(entry->second);
                    parent[std::max(a, b)] = std::min(a, b);
                }
            };
            if (domain == Element::Face)
                for (const auto h : mesh.fh_range(he::FH{handle})) {
                    const auto edge = *mesh.GetEdge(h);
                    covered_edges.insert(edge);
                    join(edge);
                }
            else {
                const auto h = mesh.GetHalfedge(he::EH{handle}, 0u);
                join(*mesh.GetFromVertex(h));
                join(*mesh.GetToVertex(h));
            }
        });
        std::unordered_map<uint32_t, uint32_t> components;
        for (uint32_t i = 0u; i < elements.size(); ++i) {
            const auto [entry, inserted] = components.try_emplace(root(i), uint32_t(groups.size()));
            if (inserted) groups.emplace_back();
            auto &group = groups[entry->second];
            if (domain == Element::Face) {
                group.Faces.push_back(elements[i]);
                for (const auto v : mesh.fv_range(he::FH{elements[i]})) group.Vertices.push_back(*v);
            } else {
                const auto h = mesh.GetHalfedge(he::EH{elements[i]}, 0u);
                group.Vertices.push_back(*mesh.GetFromVertex(h));
                group.Vertices.push_back(*mesh.GetToVertex(h));
            }
        }
    }
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
            const he::FH face{group.Faces[i]};
            plan.Words.push_back(*face);
            plan.Words.push_back(mesh.GetValence(face));
            for (const auto v : mesh.fv_range(face)) plan.Words.push_back(ranks.at(*v));
        }
    }
    for (const auto &level : levels) {
        plan.Batches.push_back({uint32_t(plan.Groups.size()), uint32_t(level.size())});
        plan.Groups.insert(plan.Groups.end(), level.begin(), level.end());
    }
    return plan;
}
