#include "mesh/EdgeChains.h"
#include "mesh/EdgeKey.h"
#include "mesh/Mesh.h"
#include "mesh/Neighborhood.h"
#include "mesh/SelectedChains.h"

#include <ranges>
#include <unordered_set>

namespace {
// Rewrites chain inputs as ranks into Outputs and orders the chains into batches by their read and write dependencies.
void ScheduleEdgeChains(EdgeChainPlan &plan, bool move_all) {
    struct Access {
        uint32_t Rank, Read{}, Write{};
    };
    std::unordered_map<uint32_t, Access> access;
    std::vector<std::vector<EdgeChain>> levels;
    for (const auto chain : plan.Chains) {
        auto vertices = std::span{plan.Inputs}.subspan(chain.InputOffset, chain.Count);
        const auto writes = [&](auto &&visit) {
            if (move_all) {
                for (const auto v : vertices) visit(v);
            } else {
                const auto phase = std::span{plan.Phases}.subspan(chain.PhaseOffset);
                for (const auto i : phase.subspan(2u + phase[0], phase[1])) visit(vertices[i]);
            }
        };
        uint32_t depth = 0u;
        for (const auto v : vertices) {
            const auto [entry, inserted] = access.try_emplace(v, Access{uint32_t(plan.Outputs.size())});
            if (inserted) plan.Outputs.push_back(v);
            depth = std::max(depth, entry->second.Write);
        }
        writes([&](uint32_t v) { depth = std::max(depth, access[v].Read); });
        for (const auto v : vertices) access[v].Read = std::max(access[v].Read, depth + 1u);
        writes([&](uint32_t v) { access[v].Write = depth + 1u; });
        for (auto &v : vertices) v = access.at(v).Rank;
        if (levels.size() <= depth) levels.resize(depth + 1u);
        levels[depth].push_back(chain);
    }
    plan.Chains.clear();
    for (const auto &level : levels) {
        plan.Batches.push_back({uint32_t(plan.Chains.size()), uint32_t(level.size())});
        plan.Chains.insert(plan.Chains.end(), level.begin(), level.end());
    }
}

void AppendPhase(std::vector<uint32_t> &phases, std::span<const uint32_t> knots, std::span<const uint32_t> points) {
    phases.push_back(uint32_t(knots.size()));
    phases.push_back(uint32_t(points.size()));
    phases.insert(phases.end(), knots.begin(), knots.end());
    phases.insert(phases.end(), points.begin(), points.end());
}
void AppendRelaxPhases(EdgeChainPlan &plan, EdgeChain &chain) {
    chain.PhaseOffset = uint32_t(plan.Phases.size());
    std::vector<uint32_t> order(chain.Count);
    for (uint32_t i = 0u; i < chain.Count; ++i) order[i] = i;
    for (uint32_t phase = 0u; phase < 2u; ++phase) {
        std::vector<uint32_t> knots, points;
        if (!chain.Closed) {
            for (uint32_t i = 0u; i < chain.Count; ++i) {
                if (i % 2u == phase) knots.push_back(i);
                else if (i > 0u && i + 1u < chain.Count) points.push_back(i);
            }
        } else {
            const bool extend = chain.Count % 2u ? phase == 1u : phase == 0u;
            const uint32_t first = !extend && phase == 1u ? 1u : 0u;
            if (extend) {
                const auto head = order.front(), tail = order.back();
                order.insert(order.begin(), tail);
                order.push_back(head);
            }
            for (uint32_t i = first; i < order.size(); i += 2u) knots.push_back(order[i]);
            for (uint32_t i = first + 1u; i < order.size(); i += 2u) {
                if (points.empty() || order[i] != points.front()) points.push_back(order[i]);
            }
            if (knots.front() != knots.back()) knots.push_back(knots.front());
        }
        AppendPhase(plan.Phases, knots, points);
    }
}
} // namespace

EdgeChainPlan PlanSelectedEdgeChains(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const GeometrySelection &selection, bool relax) {
    EdgeChainPlan plan;
    for (const auto &[pairs, closed] : SelectedEdgeChains(meshes, pipelines, mesh.GetStoreId(), selection.Edges, false).Chains) {
        if (pairs.size() < 3u) continue;
        const auto chain = VerticesOf(pairs);
        plan.Chains.push_back({uint32_t(plan.Inputs.size()), uint32_t(plan.Outputs.size()), uint32_t(chain.size()), uint32_t(closed)});
        if (relax) AppendRelaxPhases(plan, plan.Chains.back());
        plan.Inputs.insert(plan.Inputs.end(), chain.begin(), chain.end());
        plan.Outputs.insert(plan.Outputs.end(), chain.begin() + (closed ? 0u : 1u), chain.end() - (closed ? 0u : 1u));
    }
    return plan;
}

EdgeChainPlan PlanCurveBetweenSelected(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const GeometrySelection &selection, const GeometrySelection &excluded, bool extend) {
    const auto is_selected = [&](uint32_t v) { return std::ranges::binary_search(selection.Vertices, v); };
    const LocalMesh local{meshes, pipelines, mesh.GetStoreId(), HandleList(mesh.vertices())};
    struct Node {
        std::vector<uint32_t> Edges;
        std::unordered_set<uint64_t> QuadPairs;
        std::array<uint32_t, 2> Counts{};
        bool Surface{};
    };
    const auto edge_at = [&](uint32_t v, uint32_t e) -> const VertexEdge & { return *std::ranges::find(local.Edges(v), e, &VertexEdge::Edge); };
    std::unordered_map<uint32_t, Node> nodes;
    const auto node = [&](uint32_t v) -> const Node & {
        const auto [entry, inserted] = nodes.try_emplace(v);
        if (inserted) {
            auto &out = entry->second;
            for (const auto &edge : local.Edges(v)) {
                if (!std::ranges::binary_search(excluded.Edges, edge.Edge)) {
                    out.Edges.push_back(edge.Edge);
                    ++out.Counts[edge.Face == InvalidOffset];
                }
            }
            std::ranges::sort(out.Edges);
            for (const auto &item : local.Fan(v)) {
                if (item.Face == InvalidOffset) continue;
                out.Surface = true;
                const auto &corner = local.At(item.Corner);
                if (item.Valence == 4u) out.QuadPairs.insert(EdgeKey(corner.Edge, local.At(corner.Next).Edge));
            }
        }
        return entry->second;
    };
    struct Path {
        std::vector<uint32_t> Vertices;
        bool Closed{};
    };
    std::unordered_set<uint32_t> visited;
    const auto trace = [&](uint32_t start, uint32_t edge) {
        Path path;
        uint32_t at = start;
        visited.insert(edge);
        for (;;) {
            const auto &incident = edge_at(at, edge);
            at = incident.Other;
            if (at == start) {
                path.Closed = true;
                break;
            }
            path.Vertices.push_back(at);
            const bool is_wire = incident.Face == InvalidOffset;
            const auto &adjacent = node(at);
            const auto count = adjacent.Counts[is_wire];
            if (is_wire ? count != 2u : count < 3u || count > 4u) break;
            uint32_t next = InvalidOffset;
            for (const auto e : adjacent.Edges) {
                if ((edge_at(at, e).Face == InvalidOffset) == is_wire && e != edge && !visited.contains(e) && !adjacent.QuadPairs.contains(EdgeKey(edge, e))) {
                    next = e;
                    break;
                }
            }
            if (next == InvalidOffset) break;
            edge = next;
            visited.insert(edge);
        }
        return path;
    };
    const auto build = [&](uint32_t v, uint32_t e) {
        const auto &first = local.At(edge_at(v, e).Halfedge);
        const auto a = first.From, b = first.Vertex;
        auto forward = trace(a, e);
        if (forward.Closed) forward.Vertices.insert(forward.Vertices.begin(), a);
        else {
            auto backward = trace(b, e);
            std::ranges::reverse(backward.Vertices);
            forward.Vertices.insert(forward.Vertices.begin(), backward.Vertices.begin(), backward.Vertices.end());
        }
        return forward;
    };
    std::vector<Path> paths;
    for (const auto v : selection.Vertices) {
        for (const auto e : node(v).Edges) {
            if (visited.contains(e)) continue;
            auto path = build(v, e);
            if (path.Vertices.size() < 3u) continue;
            const auto count = size_t(std::ranges::count_if(path.Vertices, is_selected));
            if (count == path.Vertices.size()) {
                for (const auto x : path.Vertices) {
                    for (const auto other : node(x).Edges) {
                        if (!visited.contains(other)) paths.push_back(build(x, other));
                    }
                }
            } else if (count) {
                paths.push_back(std::move(path));
            }
        }
    }
    EdgeChainPlan plan;
    for (auto &path : paths) {
        auto &vertices = path.Vertices;
        const auto n = uint32_t(vertices.size());
        if (n < 3u) continue;
        std::vector<uint32_t> selected;
        for (uint32_t i = 0u; i < n; ++i) {
            if (is_selected(vertices[i])) selected.push_back(i);
        }
        if (selected.empty()) continue;
        if (path.Closed) {
            uint32_t gap = 0u, gap_start = 0u;
            for (uint32_t i = 0u; i < selected.size(); ++i) {
                const auto size = (selected[(i + 1u) % selected.size()] + n - selected[i] - 1u) % n;
                if (size > gap) {
                    gap = size;
                    gap_start = (selected[i] + 1u) % n;
                }
            }
            if (!extend) {
                if (gap) std::ranges::rotate(vertices, vertices.begin() + (gap_start + gap) % n);
                path.Closed = false;
            } else if (gap > 2u * (n / 4u)) {
                const auto first = (gap_start + gap + n - n / 4u) % n, last = (gap_start + n - 1u + n / 4u) % n;
                std::ranges::rotate(vertices, vertices.begin() + first);
                vertices.resize((last + n - first) % n + 1u);
                path.Closed = false;
            } else {
                std::ranges::rotate(vertices, vertices.begin() + selected.front());
            }
        }
        if (!extend) {
            const auto first = std::ranges::find_if(vertices, is_selected), last = std::ranges::find_if(std::views::reverse(vertices), is_selected).base();
            vertices = std::vector<uint32_t>(first, last);
        }
        if (vertices.size() < 3u) continue;
        std::vector<uint32_t> knots, points;
        for (uint32_t i = 0u; i < vertices.size(); ++i) {
            if (is_selected(vertices[i]) || (!path.Closed && (i == 0u || i + 1u == vertices.size()))) knots.push_back(i);
            else points.push_back(i);
        }
        if (points.empty()) continue;
        const EdgeChain chain{.InputOffset = uint32_t(plan.Inputs.size()), .Count = uint32_t(vertices.size()), .Closed = uint32_t(path.Closed), .PhaseOffset = uint32_t(plan.Phases.size())};
        AppendPhase(plan.Phases, knots, points);
        for (const auto p : points) plan.Phases.push_back(uint32_t(node(vertices[p]).Surface));
        plan.Inputs.insert(plan.Inputs.end(), vertices.begin(), vertices.end());
        plan.Chains.push_back(chain);
    }
    ScheduleEdgeChains(plan, false);
    return plan;
}

EdgeChainPlan PlanCircularize(const MeshStore &meshes, const MeshPipelines &pipelines, const Mesh &mesh, const GeometrySelection &selection, const GeometrySelection &excluded) {
    std::vector<uint32_t> faces;
    for (const auto f : selection.Faces) {
        if (!std::ranges::binary_search(excluded.Faces, f)) faces.push_back(f);
    }
    const auto facts = SelectedEdgeFacts(meshes, pipelines, mesh.GetStoreId(), selection.Edges, faces);
    std::array<std::vector<uint32_t>, 2> graphs;
    std::unordered_set<uint32_t> surface_vertices;
    for (uint32_t i = 0u; i < facts.size(); ++i) {
        const auto e = selection.Edges[i];
        const auto &[a, b, source, face, selected_faces] = facts[i];
        const bool wire = face == InvalidOffset;
        if (!wire) {
            surface_vertices.insert(a);
            surface_vertices.insert(b);
        }
        if (std::ranges::binary_search(excluded.Edges, e) || a == b || selected_faces > 1u) continue;
        graphs[wire].push_back(e);
    }
    EdgeChainPlan plan;
    for (uint32_t wire = 0u; wire < 2u; ++wire) {
        for (const auto &[pairs, closed] : SelectedEdgeChains(meshes, pipelines, mesh.GetStoreId(), graphs[wire], false).Chains) {
            if (pairs.size() < 3u) continue;
            if (wire && std::ranges::any_of(pairs, [&](const auto &pair) { return surface_vertices.contains(pair[0]); })) continue;
            plan.Chains.push_back({.InputOffset = uint32_t(plan.Inputs.size()), .Count = uint32_t(pairs.size()), .Closed = uint32_t(closed)});
            for (const auto &[vertex, source] : pairs) plan.Inputs.push_back(vertex);
        }
    }
    ScheduleEdgeChains(plan, true);
    return plan;
}
