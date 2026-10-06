#include "mesh/EdgeChains.h"
#include "SortUnique.h"
#include "mesh/Mesh.h"
#include "mesh/MeshEdgeUsers.h"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <unordered_set>

void VisitEdgeChains(EdgeGraph &graph, const std::function<void(const std::vector<uint32_t> &, bool)> &visit) {
    std::vector<uint32_t> vertices;
    for (auto &[v, adjacent] : graph) {
        SortUnique(adjacent);
        vertices.push_back(v);
    }
    std::ranges::sort(vertices);
    std::unordered_set<uint64_t> used;
    const auto walk = [&](uint32_t start, uint32_t next) {
        if (!used.insert(MeshEdgeUsers::Key(start, next)).second) return;
        std::vector<uint32_t> chain{start};
        uint32_t previous = start;
        bool closed = false;
        for (;;) {
            // Blender treats a loop returning to a junction as an open chain
            // after removing its repeated endpoint.
            if (next == start) {
                closed = graph.at(start).size() == 2u;
                break;
            }
            chain.push_back(next);
            const auto &adjacent = graph.at(next);
            if (adjacent.size() != 2u) break;
            const auto following = adjacent[0] == previous ? adjacent[1] : adjacent[0];
            if (!used.insert(MeshEdgeUsers::Key(next, following)).second) break;
            previous = next;
            next = following;
        }
        visit(chain, closed);
    };
    // Endpoints and junctions delimit open chains. Remaining edges form cycles.
    for (const auto v : vertices)
        if (graph.at(v).size() != 2u)
            for (const auto next : graph.at(v)) walk(v, next);
    for (const auto v : vertices)
        for (const auto next : graph.at(v)) walk(v, next);
}

namespace {
// Convert canonical path handles to a shared position scratch and order read/write
// dependencies. Paths sharing only fixed controls can execute concurrently.
void ScheduleEdgeChains(EdgeChainPlan &plan, bool move_all) {
    struct Access {
        uint32_t Rank, Read{}, Write{};
    };
    std::unordered_map<uint32_t, Access> access;
    std::vector<std::vector<EdgeChain>> levels;
    for (const auto chain : plan.Chains) {
        auto vertices = std::span{plan.Inputs}.subspan(chain.InputOffset, chain.Count);
        const auto writes = [&](auto &&visit) {
            if (move_all)
                for (const auto v : vertices) visit(v);
            else {
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

// Blender's alternating phases include wrapped knots for closed loops.
// Phase records are (knot count, point count, knot indices, point indices).
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
            for (uint32_t i = first + 1u; i < order.size(); i += 2u)
                if (points.empty() || order[i] != points.front()) points.push_back(order[i]);
            if (knots.front() != knots.back()) knots.push_back(knots.front());
        }
        plan.Phases.push_back(uint32_t(knots.size()));
        plan.Phases.push_back(uint32_t(points.size()));
        plan.Phases.insert(plan.Phases.end(), knots.begin(), knots.end());
        plan.Phases.insert(plan.Phases.end(), points.begin(), points.end());
    }
}
} // namespace

EdgeChainPlan PlanSelectedEdgeChains(const Mesh &mesh, const GeometrySelection &selection, bool relax) {
    EdgeGraph neighbors;
    std::ranges::for_each(selection.Edges, [&](uint32_t edge) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
        const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
        if (a == b) return;
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
    });
    EdgeChainPlan plan;
    VisitEdgeChains(neighbors, [&](const auto &chain, bool closed) {
        if (chain.size() < 3u) return;
        plan.Chains.push_back({uint32_t(plan.Inputs.size()), uint32_t(plan.Outputs.size()), uint32_t(chain.size()), uint32_t(closed)});
        if (relax) AppendRelaxPhases(plan, plan.Chains.back());
        plan.Inputs.insert(plan.Inputs.end(), chain.begin(), chain.end());
        plan.Outputs.insert(plan.Outputs.end(), chain.begin() + (closed ? 0u : 1u), chain.end() - (closed ? 0u : 1u));
    });
    return plan;
}

EdgeChainPlan PlanCurveBetweenSelected(const Mesh &mesh, const GeometrySelection &selection, const GeometrySelection &excluded, bool extend) {
    const auto incidence = mesh.GetVertexEdgeIncidence();
    const auto &connectivity = mesh.GetConnectivity();
    const auto wire = [&](uint32_t e) { return !mesh.GetFace(mesh.GetHalfedge(he::EH{e}, 0u)); };
    struct Node {
        std::vector<uint32_t> Edges;
        std::unordered_set<uint64_t> QuadPairs;
        std::array<uint32_t, 2> Counts{};
        bool Surface{};
    };
    std::unordered_map<uint32_t, Node> nodes;
    const auto node = [&](uint32_t v) -> const Node & {
        const auto [entry, inserted] = nodes.try_emplace(v);
        if (inserted) {
            auto &out = entry->second;
            for (const auto e : incidence.Incident(v))
                if (!std::ranges::binary_search(excluded.Edges, e)) {
                    out.Edges.push_back(e);
                    ++out.Counts[wire(e)];
                }
            std::ranges::sort(out.Edges);
            const auto fan = connectivity.VertexCorners[v];
            for (uint32_t i = 0u; i < fan.y; ++i) {
                const he::HH h{connectivity.FanItems[fan.x + i].x};
                const auto face = mesh.GetFace(h);
                if (!face) continue;
                out.Surface = true;
                if (mesh.GetValence(face) == 4u) out.QuadPairs.insert(MeshEdgeUsers::Key(*mesh.GetEdge(h), *mesh.GetEdge(connectivity.Next(h))));
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
            const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
            const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
            at = a == at ? b : a;
            if (at == start) {
                path.Closed = true;
                break;
            }
            path.Vertices.push_back(at);
            const bool is_wire = wire(edge);
            const auto &adjacent = node(at);
            const auto count = adjacent.Counts[is_wire];
            if (is_wire ? count != 2u : count < 3u || count > 4u) break;
            uint32_t next = InvalidOffset;
            for (const auto e : adjacent.Edges)
                if (wire(e) == is_wire && e != edge && !visited.contains(e) && !adjacent.QuadPairs.contains(MeshEdgeUsers::Key(edge, e))) {
                    next = e;
                    break;
                }
            if (next == InvalidOffset) break;
            edge = next;
            visited.insert(edge);
        }
        return path;
    };
    const auto build = [&](uint32_t e) {
        const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
        const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
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
    std::ranges::for_each(selection.Vertices, [&](uint32_t v) {
        for (const auto e : node(v).Edges) {
            if (visited.contains(e)) continue;
            auto path = build(e);
            if (path.Vertices.size() < 3u) continue;
            const auto count = size_t(std::ranges::count_if(path.Vertices, [&](uint32_t x) { return std::ranges::binary_search(selection.Vertices, x); }));
            if (count == path.Vertices.size()) {
                for (const auto x : path.Vertices)
                    for (const auto other : node(x).Edges)
                        if (!visited.contains(other)) paths.push_back(build(other));
            } else if (count) paths.push_back(std::move(path));
        }
    });
    EdgeChainPlan plan;
    for (auto &path : paths) {
        auto &vertices = path.Vertices;
        const auto n = uint32_t(vertices.size());
        if (n < 3u) continue;
        std::vector<uint32_t> selected;
        for (uint32_t i = 0u; i < n; ++i)
            if (std::ranges::binary_search(selection.Vertices, vertices[i])) selected.push_back(i);
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
                if (gap) std::rotate(vertices.begin(), vertices.begin() + (gap_start + gap) % n, vertices.end());
                path.Closed = false;
            } else if (gap > 2u * (n / 4u)) {
                const auto first = (gap_start + gap + n - n / 4u) % n, last = (gap_start + n - 1u + n / 4u) % n;
                std::rotate(vertices.begin(), vertices.begin() + first, vertices.end());
                vertices.resize((last + n - first) % n + 1u);
                path.Closed = false;
            } else std::rotate(vertices.begin(), vertices.begin() + selected.front(), vertices.end());
        }
        if (!extend) {
            const auto is_selected = [&](uint32_t v) { return std::ranges::binary_search(selection.Vertices, v); };
            const auto first = std::ranges::find_if(vertices, is_selected), last = std::find_if(vertices.rbegin(), vertices.rend(), is_selected).base();
            vertices = std::vector<uint32_t>(first, last);
        }
        if (vertices.size() < 3u) continue;
        std::vector<uint32_t> knots, points;
        for (uint32_t i = 0u; i < vertices.size(); ++i) {
            if (std::ranges::binary_search(selection.Vertices, vertices[i]) || (!path.Closed && (i == 0u || i + 1u == vertices.size()))) knots.push_back(i);
            else points.push_back(i);
        }
        if (points.empty()) continue;
        EdgeChain chain{.InputOffset = uint32_t(plan.Inputs.size()), .Count = uint32_t(vertices.size()), .Closed = uint32_t(path.Closed), .PhaseOffset = uint32_t(plan.Phases.size())};
        plan.Phases.push_back(uint32_t(knots.size()));
        plan.Phases.push_back(uint32_t(points.size()));
        plan.Phases.insert(plan.Phases.end(), knots.begin(), knots.end());
        plan.Phases.insert(plan.Phases.end(), points.begin(), points.end());
        for (const auto p : points) plan.Phases.push_back(uint32_t(node(vertices[p]).Surface));
        plan.Inputs.insert(plan.Inputs.end(), vertices.begin(), vertices.end());
        plan.Chains.push_back(chain);
    }
    ScheduleEdgeChains(plan, false);
    return plan;
}

EdgeChainPlan PlanCircularize(const Mesh &mesh, const GeometrySelection &selection, const GeometrySelection &excluded) {
    std::unordered_map<uint32_t, uint32_t> selected_faces;
    std::ranges::for_each(selection.Faces, [&](uint32_t f) {
        if (!std::ranges::binary_search(excluded.Faces, f))
            for (const auto h : mesh.fh_range(he::FH{f})) ++selected_faces[*mesh.GetEdge(h)];
    });
    std::array<EdgeGraph, 2> graphs;
    std::unordered_set<uint32_t> surface_vertices;
    std::ranges::for_each(selection.Edges, [&](uint32_t e) {
        const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
        const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h);
        const bool wire = !mesh.GetFace(h);
        if (!wire) {
            surface_vertices.insert(a);
            surface_vertices.insert(b);
        }
        if (std::ranges::binary_search(excluded.Edges, e) || a == b || selected_faces[e] > 1u) return;
        graphs[wire][a].push_back(b);
        graphs[wire][b].push_back(a);
    });
    EdgeChainPlan plan;
    for (uint32_t wire = 0u; wire < 2u; ++wire) {
        auto &graph = graphs[wire];
        VisitEdgeChains(graph, [&](const auto &path, bool closed) {
            if (path.size() < 3u) return;
            if (wire && std::ranges::any_of(path, [&](uint32_t v) { return surface_vertices.contains(v); })) return;
            plan.Chains.push_back({.InputOffset = uint32_t(plan.Inputs.size()), .Count = uint32_t(path.size()), .Closed = uint32_t(closed)});
            plan.Inputs.insert(plan.Inputs.end(), path.begin(), path.end());
        });
    }
    ScheduleEdgeChains(plan, true);
    return plan;
}
