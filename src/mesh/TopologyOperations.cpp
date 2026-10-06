#include "mesh/TopologyOperations.h"
#include "mesh/EdgeChains.h"
#include "mesh/Mesh.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshStore.h"
#include "numeric/MatrixMath.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <unordered_set>

namespace {
bool Finite(vec2 value) { return std::isfinite(value.x) && std::isfinite(value.y); }
bool Finite(vec3 value) { return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z); }
bool FiniteMatrix(const auto &value, uint32_t size) {
    for (uint32_t column = 0u; column < size; ++column)
        for (uint32_t row = 0u; row < size; ++row)
            if (!std::isfinite(value[column][row])) return false;
    return true;
}
std::vector<uint32_t> EdgeRing(const Mesh &mesh, uint32_t edge) {
    std::vector<uint32_t> ring{edge};
    std::unordered_set<uint32_t> visited{edge};
    const auto &c = mesh.GetConnectivity();
    const auto start = mesh.GetHalfedge(he::EH{edge}, 0);
    for (const auto side : {start, c.Opposites[*start]}) {
        auto h = side;
        while (h) {
            const auto face = c.FaceOf(h);
            if (!face || mesh.GetValence(face) != 4) break;
            const auto across = c.Next(c.Next(h));
            const auto e = mesh.GetEdge(across);
            if (!visited.insert(*e).second) break;
            ring.push_back(*e);
            h = c.Opposites[*across];
        }
    }
    return ring;
}

std::vector<std::vector<uint32_t>> BoundaryLoops(const MeshStore &meshes, const Mesh &mesh, std::span<const uint32_t> selected_edges, bool selected_only, uint32_t max_sides = 0u) {
    const auto &c = mesh.GetConnectivity();
    // Both views visit their edges in ascending handle order.
    std::vector<uint32_t> edges;
    const auto collect = [&](const auto &view) { view.ForEach([&](uint32_t edge) { edges.push_back(edge); }); };
    if (selected_only) edges.assign(selected_edges.begin(), selected_edges.end());
    else collect(meshes.GetBoundaryEdges(mesh.GetStoreId()));
    std::vector<uint32_t> starts;
    for (const auto edge : edges) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
        if (!c.Opposites[*h]) starts.push_back(*h);
    }
    std::ranges::sort(starts);
    const auto candidate = [&](uint32_t h) {
        return h != InvalidOffset && !c.Opposites[h] && std::ranges::binary_search(edges, *mesh.GetEdge(Mesh::HH{h}));
    };
    std::unordered_map<uint32_t, uint32_t> selected_outgoing;
    if (selected_only)
        for (const auto h : starts) {
            const auto vertex = *mesh.GetFromVertex(Mesh::HH{h});
            const auto [it, unique] = selected_outgoing.emplace(vertex, h);
            if (!unique) it->second = InvalidOffset;
        }
    // Follow the face fan at the current boundary halfedge's destination to
    // find the next boundary halfedge on the same surface sheet. Vertex-based
    // pairing loses loops when distinct boundaries share a vertex.
    const auto successor = [&](uint32_t h) -> uint32_t {
        auto next = c.Next(Mesh::HH{h});
        const auto across = [&](Mesh::HH at) -> Mesh::HH {
            if (!at) return {};
            const auto opposite = c.Opposites[*at];
            return opposite ? c.Next(opposite) : Mesh::HH{};
        };
        auto fast = next;
        while (next && c.Opposites[*next]) {
            next = across(next);
            fast = across(across(fast));
            if (fast && next == fast) return InvalidOffset;
        }
        return next ? *next : InvalidOffset;
    };
    std::vector<std::vector<uint32_t>> loops;
    std::unordered_set<uint32_t> used;
    used.reserve(starts.size());
    for (const auto start : starts) {
        if (used.contains(start)) continue;
        std::vector<uint32_t> loop;
        auto h = start;
        bool closed = false;
        uint32_t length = 0u;
        while (candidate(h)) {
            if (!used.insert(h).second) {
                closed = h == start;
                break;
            }
            ++length;
            if (!max_sides || loop.size() < max_sides) loop.push_back(*mesh.GetFromVertex(Mesh::HH{h}));
            const auto next = successor(h);
            if (selected_only && !candidate(next)) {
                // A selected hole may touch an unselected boundary at one
                // vertex. Follow its sole selected outgoing edge there.
                const auto it = selected_outgoing.find(*mesh.GetToVertex(Mesh::HH{h}));
                h = it == selected_outgoing.end() ? InvalidOffset : it->second;
            } else h = next;
        }
        if (closed && length >= 3u && (!max_sides || length <= max_sides)) {
            std::ranges::reverse(loop);
            loops.push_back(std::move(loop));
        }
    }
    return loops;
}

MeshTopologyTask PrimitiveListTask(const MeshStore &meshes, const Mesh &mesh, std::span<const std::vector<uint32_t>> loops, const std::unordered_map<uint64_t, uint32_t> &edge_sources = {}, std::span<const uint32_t> grid_loop = {}, uint32_t grid_span = 0u) {
    const auto appended_base = meshes.Arenas().Vertices.Capacity();
    const uint64_t vertices = grid_loop.empty() ? 0u : uint64_t(grid_span - 1u) * (grid_loop.size() / 2u - grid_span - 1u);
    if (uint64_t(appended_base) + vertices > UINT32_MAX) throw std::length_error("Face list exceeds the vertex handle address space.");
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::AddPrimitives, .AppendedBase = appended_base};
    uint32_t attribute_source = InvalidOffset;
    for (const auto &loop : loops)
        for (const auto vertex : loop)
            if (vertex < appended_base && attribute_source == InvalidOffset) attribute_source = vertex;
    if (attribute_source == InvalidOffset) throw std::invalid_argument("Face creation needs a source vertex for attributes.");
    task.List = {uint32_t(vertices), uint32_t(grid_loop.size()), grid_span, attribute_source};
    task.List.insert(task.List.end(), grid_loop.begin(), grid_loop.end());
    task.List.push_back(uint32_t(loops.size()));
    for (const auto &loop : loops) {
        task.List.push_back(uint32_t(loop.size()));
        for (uint32_t i = 0u; i < loop.size(); ++i) {
            task.List.push_back(loop[i]);
            const auto edge = edge_sources.find(MeshEdgeUsers::Key(loop[(i + loop.size() - 1u) % loop.size()], loop[i]));
            task.List.push_back(edge == edge_sources.end() ? InvalidOffset : edge->second);
        }
    }
    return task;
}

std::unordered_map<uint64_t, uint32_t> SelectedEdgeSources(const Mesh &mesh, std::span<const uint32_t> selected_edges, bool boundary_only) {
    std::unordered_map<uint64_t, uint32_t> sources;
    MeshEdgeUsers users{mesh};
    std::ranges::for_each(selected_edges, [&](uint32_t edge) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
        if (mesh.GetFromVertex(h) == mesh.GetToVertex(h)) return;
        const auto key = users.Key(h);
        // Retain the selected surface corner even when a coincident wire supplied a different radial corner first.
        if (!boundary_only && mesh.GetConnectivity().FaceOf(h)) {
            sources[key] = *h;
            return;
        }
        const auto adjacent = users.Get(h);
        if (!boundary_only || adjacent.Count <= 1u) sources.try_emplace(key, adjacent.Count ? adjacent.First : *h);
    });
    return sources;
}

struct SelectedChain {
    std::vector<uint32_t> Vertices;
    bool Closed{};
    int Winding{};
};

std::vector<SelectedChain> SelectedChains(const Mesh &mesh, std::span<const uint32_t> selected_edges, std::unordered_map<uint64_t, uint32_t> &sources) {
    sources = SelectedEdgeSources(mesh, selected_edges, true);
    const auto &c = mesh.GetConnectivity();
    EdgeGraph neighbors;
    for (const auto &[key, h] : sources) {
        const auto a = uint32_t(key >> 32u), b = uint32_t(key);
        neighbors[a].push_back(b);
        neighbors[b].push_back(a);
    }
    for (const auto &[v, adjacent] : neighbors)
        if (adjacent.size() > 2u) return {};
    std::vector<SelectedChain> chains;
    VisitEdgeChains(neighbors, [&](const auto &vertices, bool closed) {
        auto &chain = chains.emplace_back(SelectedChain{vertices, closed});
        for (size_t i = 0u; i < vertices.size() - (closed ? 0u : 1u); ++i) {
            const auto v = vertices[i], next = vertices[(i + 1u) % vertices.size()];
            const auto h = he::HH{sources.at(MeshEdgeUsers::Key(v, next))};
            if (c.FaceOf(h)) chain.Winding += *mesh.GetFromVertex(h) == v ? 1 : -1;
        }
    });
    return chains;
}
} // namespace

std::vector<MeshTopologyTask> SeparateGeometryTasks(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection, GeometrySeparateMode mode) {
    mesh.ValidateSelection(selection);
    if (uint32_t(mode) > uint32_t(GeometrySeparateMode::Material)) throw std::invalid_argument("Invalid separate mode.");
    const auto id = mesh.GetStoreId();
    std::vector<MeshTopologyTask> tasks;
    Element element = Element::Vertex;
    std::vector<std::vector<uint32_t>> groups;
    if (mode == GeometrySeparateMode::Selected) {
        element = mesh.FaceCount() ? Element::Face : Element::Vertex;
        auto &group = groups.emplace_back();
        group = selection.Get(element);
        if (group.empty()) groups.clear();
    } else if (mode == GeometrySeparateMode::LooseParts) {
        const auto incidence = mesh.GetVertexEdgeIncidence();
        std::unordered_set<uint32_t> visited;
        for (const auto v : mesh.vertices()) {
            if (!visited.insert(*v).second) continue;
            auto &group = groups.emplace_back(1u, *v);
            for (size_t at = 0u; at < group.size(); ++at)
                for (const auto edge : incidence.Incident(group[at])) {
                    const auto h = mesh.GetHalfedge(he::EH{edge}, 0u);
                    const auto a = *mesh.GetFromVertex(h), b = *mesh.GetToVertex(h), other = a == group[at] ? b : a;
                    if (visited.insert(other).second) group.push_back(other);
                }
        }
        // Blender keeps the first connected component in the original object.
        if (!groups.empty()) groups.erase(groups.begin());
    } else {
        element = Element::Face;
        const auto &a = meshes.Arenas();
        const auto palette = a.PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
        std::unordered_map<uint32_t, uint32_t> materials;
        for (const auto face : mesh.faces()) {
            const auto material = palette[a.FacePrimitives.Get(*face)];
            const auto [entry, inserted] = materials.try_emplace(material, uint32_t(groups.size()));
            if (inserted) groups.emplace_back();
            groups[entry->second].push_back(*face);
        }
        // Blender extracts successive material groups, leaving the last in place.
        if (!groups.empty()) groups.pop_back();
    }
    if (groups.empty()) return {};
    std::vector<uint32_t> removed;
    for (auto &group : groups) {
        std::ranges::sort(group);
        removed.insert(removed.end(), group.begin(), group.end());
        tasks.push_back({.SourceId = id, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = element});
        (element == Element::Face ? tasks.back().Selection.Faces : tasks.back().Selection.Vertices) = std::move(group);
    }
    std::ranges::sort(removed);
    tasks.push_back({.SourceId = id, .Op = element == Element::Face ? MeshTopologyOp::DeleteFaces : mesh.FaceCount() ? MeshTopologyOp::DeleteVertices :
                                                                                                                       MeshTopologyOp::DeleteEdges,
                     .SelectionElement = element,
                     .Selection = {}});
    (element == Element::Face ? tasks.back().Selection.Faces : tasks.back().Selection.Vertices) = std::move(removed);
    return tasks;
}

std::optional<MeshTopologyTask> BridgeEdgeLoopsTask(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) {
    mesh.ValidateSelection(selection);
    std::unordered_map<uint64_t, uint32_t> sources;
    auto chains = SelectedChains(mesh, selection.Edges, sources);
    if (chains.size() != 2u || chains[0].Closed != chains[1].Closed) return {};
    if (chains[0].Vertices.size() < chains[1].Vertices.size()) std::swap(chains[0], chains[1]);
    const bool closed = chains[0].Closed;
    auto &a = chains[0].Vertices, &b = chains[1].Vertices;
    const auto flip = [&](auto &chain) { std::reverse(chain.begin() + (closed ? 1u : 0u), chain.end()); };
    // New faces oppose the existing face along each rail.
    if (chains[0].Winding < 0) flip(a);
    if (chains[1].Winding > 0) flip(b);
    const auto position = [&](uint32_t v) { return mesh.GetPosition(he::VH{v}); };
    const auto normal = [&](const auto &loop) {
        vec3 n{};
        const auto origin = position(loop[0]);
        for (uint32_t i = 1u; i + 1u < loop.size(); ++i) n += Cross(position(loop[i]) - origin, position(loop[i + 1u]) - origin);
        return n;
    };
    const auto align = [&](const auto &fixed, auto &free) {
        if (closed) {
            if (Dot(normal(fixed), normal(free)) < 0.f) flip(free);
        } else {
            const auto same = Distance2(position(fixed.front()), position(free.front())) + Distance2(position(fixed.back()), position(free.back()));
            const auto crossed = Distance2(position(fixed.front()), position(free.back())) + Distance2(position(fixed.back()), position(free.front()));
            if (crossed < same) flip(free);
        }
    };
    if (!chains[1].Winding) align(a, b);
    else if (!chains[0].Winding) align(b, a);
    if (closed) {
        if (!chains[0].Winding && !chains[1].Winding) {
            vec3 separation{};
            for (const auto v : a) separation += position(v) / float(a.size());
            for (const auto v : b) separation -= position(v) / float(b.size());
            if (Dot(normal(a), separation) < 0.f) {
                flip(a);
                flip(b);
            }
        }
        uint32_t start = 0u;
        float best = std::numeric_limits<float>::max();
        for (uint32_t j = 0u; j < b.size(); ++j)
            if (const auto distance = Distance2(position(a[0]), position(b[j])); distance < best) {
                best = distance;
                start = j;
            }
        std::rotate(b.begin(), b.begin() + start, b.end());
    }
    const auto na = uint32_t(a.size()) - uint32_t(!closed), nb = uint32_t(b.size()) - uint32_t(!closed);
    std::vector<std::vector<uint32_t>> faces;
    faces.reserve(na);
    for (uint32_t i = 0u; i < na; ++i) {
        const auto j0 = uint32_t(uint64_t(i) * nb / na), j1 = uint32_t(uint64_t(i + 1u) * nb / na);
        const auto next = a[(i + 1u) % a.size()];
        if (j0 == j1) faces.push_back({next, a[i], b[j0]});
        else faces.push_back({next, a[i], b[j0], b[j1 % b.size()]});
    }
    return PrimitiveListTask(meshes, mesh, faces, sources);
}

std::optional<MeshTopologyTask> GridFillTask(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection, uint32_t span) {
    mesh.ValidateSelection(selection);
    std::unordered_map<uint64_t, uint32_t> sources;
    auto chains = SelectedChains(mesh, selection.Edges, sources);
    if (chains.size() != 1u || !chains[0].Closed || chains[0].Vertices.size() % 2u || chains[0].Vertices.size() < 4u) return {};
    auto &loop = chains[0].Vertices;
    if (chains[0].Winding < 0) std::reverse(loop.begin() + 1u, loop.end());
    const auto length = uint32_t(loop.size());
    const auto appended_base = meshes.Arenas().Vertices.Capacity();
    const uint32_t s = std::clamp(span == 0 ? std::max(length / 4, 1u) : span, 1u, length / 2 - 1), t = length / 2 - s;
    const uint64_t interior = uint64_t(s - 1u) * (t - 1u), cells = uint64_t(s) * t;
    if (uint64_t(appended_base) + interior > UINT32_MAX || 5ull + length + 9u * cells > UINT32_MAX) {
        throw std::length_error("Grid fill exceeds the vertex or face-list address space.");
    }
    // Nodes run along the first side (u) and up the second (v), with the loop's four sides as the rails.
    const auto node = [&](uint32_t i, uint32_t j) {
        if (j == 0u) return loop[i];
        if (j == t) return loop[(2u * s + t - i) % length];
        if (i == 0u) return loop[(length - j) % length];
        if (i == s) return loop[s + j];
        return appended_base + (j - 1u) * (s - 1u) + i - 1u;
    };
    std::vector<std::vector<uint32_t>> faces;
    for (uint32_t j = 0; j < t; ++j) {
        for (uint32_t i = 0; i < s; ++i) faces.push_back({node(i + 1, j), node(i, j), node(i, j + 1), node(i + 1, j + 1)});
    }
    return PrimitiveListTask(meshes, mesh, faces, sources, loop, s);
}

std::optional<MeshTopologyTask> FillHolesTask(const MeshStore &meshes, const Mesh &mesh, uint32_t sides) {
    mesh.ValidateSelection({});
    auto loops = BoundaryLoops(meshes, mesh, {}, false, sides);
    if (loops.empty()) return {};
    return PrimitiveListTask(meshes, mesh, loops);
}

std::optional<MeshTopologyTask> ConvexHullTask(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) {
    mesh.ValidateSelection(selection);
    const auto &points = selection.Vertices;
    if (points.size() < 4) return {};
    const auto at = [&](uint32_t v) { return mesh.GetPosition(he::VH{v}); };
    // A starting tetrahedron from the first point, the farthest from it, the farthest from that line, and the farthest from that plane.
    std::array<uint32_t, 4> seed{points[0], points[0], points[0], points[0]};
    const auto farthest = [&](uint32_t &vertex, auto &&distance) {
        float best = 0.f;
        for (const auto v : points)
            if (const auto d = distance(v); d > best) {
                best = d;
                vertex = v;
            }
        return best;
    };
    const float extent = std::sqrt(farthest(seed[1], [&](uint32_t v) { return Distance2(at(v), at(seed[0])); }));
    farthest(seed[2], [&](uint32_t v) { return Length(Cross(at(seed[1]) - at(seed[0]), at(v) - at(seed[0]))); });
    const auto seed_normal = Cross(at(seed[1]) - at(seed[0]), at(seed[2]) - at(seed[0]));
    if (farthest(seed[3], [&](uint32_t v) { return std::abs(Dot(seed_normal, at(v) - at(seed[0]))); }) < 1e-12f) return {};

    struct Face {
        std::array<uint32_t, 3> V;
        // The face across each edge V[k] to V[k + 1].
        std::array<uint32_t, 3> Across{};
        vec3 Normal;
        float Offset, Tolerance;
        std::vector<uint32_t> Outside;
        bool Alive{true};
    };
    std::vector<Face> faces;
    // A point counts as outside a face beyond a sliver of the point set's extent, so coplanar points stay inside.
    const float tolerance = 1e-6f * extent;
    const auto make = [&](std::array<uint32_t, 3> v) {
        Face face{.V = v, .Normal = Cross(at(v[1]) - at(v[0]), at(v[2]) - at(v[0]))};
        face.Offset = Dot(face.Normal, at(v[0]));
        face.Tolerance = Length(face.Normal) * tolerance;
        return face;
    };
    const auto height = [&](const Face &face, uint32_t p) { return Dot(face.Normal, at(p)) - face.Offset; };
    const auto outside = [&](const Face &face, uint32_t p) { return height(face, p) > face.Tolerance; };
    const auto outward = [&](std::array<uint32_t, 3> tri, uint32_t inside) {
        const auto n = Cross(at(tri[1]) - at(tri[0]), at(tri[2]) - at(tri[0]));
        return Dot(n, at(inside) - at(tri[0])) > 0.f ? std::array{tri[0], tri[2], tri[1]} : tri;
    };
    // The slot of the edge `from` to `to` on a face, or three when the face lacks it.
    const auto edge_slot = [](const Face &face, uint32_t from, uint32_t to) {
        uint32_t j = 0;
        while (j < 3 && !(face.V[j] == from && face.V[(j + 1) % 3] == to)) ++j;
        return j;
    };
    // A point waits on the first face from `first` that sees it, or falls inside.
    const auto assign = [&](uint32_t v, uint32_t first) {
        for (uint32_t i = first; i < faces.size(); ++i) {
            if (outside(faces[i], v)) {
                faces[i].Outside.push_back(v);
                return;
            }
        }
    };
    std::vector<uint32_t> pending;
    const auto enqueue = [&](uint32_t first) {
        for (uint32_t i = first; i < faces.size(); ++i)
            if (!faces[i].Outside.empty()) pending.push_back(i);
    };
    faces.push_back(make(outward({seed[0], seed[1], seed[2]}, seed[3])));
    faces.push_back(make(outward({seed[0], seed[1], seed[3]}, seed[2])));
    faces.push_back(make(outward({seed[0], seed[2], seed[3]}, seed[1])));
    faces.push_back(make(outward({seed[1], seed[2], seed[3]}, seed[0])));
    // The tetrahedron's faces meet across each shared edge, in opposite directions.
    for (uint32_t a = 0; a < 4; ++a) {
        for (uint32_t k = 0; k < 3; ++k) {
            for (uint32_t b = 0; b < 4; ++b) {
                if (edge_slot(faces[b], faces[a].V[(k + 1) % 3], faces[a].V[k]) < 3) faces[a].Across[k] = b;
            }
        }
    }
    for (const auto v : points)
        if (std::ranges::find(seed, v) == seed.end()) assign(v, 0);
    enqueue(0);

    struct HorizonEdge {
        uint32_t From, To, Neighbor, NeighborEdge;
    };
    std::vector<uint32_t> visible, stack;
    std::vector<HorizonEdge> horizon;
    // The search that last reached each face.
    std::vector<uint32_t> visited(faces.size(), 0u);
    uint32_t search = 0;
    while (!pending.empty()) {
        const auto start = pending.back();
        pending.pop_back();
        if (!faces[start].Alive || faces[start].Outside.empty()) continue;
        const auto p = *std::ranges::max_element(faces[start].Outside, {}, [&](uint32_t v) { return height(faces[start], v); });
        // The faces the point sees form one connected region, whose boundary edges are the horizon.
        visible.clear();
        horizon.clear();
        ++search;
        stack.assign(1, start);
        visited[start] = search;
        while (!stack.empty()) {
            const auto f = stack.back();
            stack.pop_back();
            visible.push_back(f);
            for (uint32_t k = 0; k < 3; ++k) {
                const auto n = faces[f].Across[k];
                const bool sees = outside(faces[n], p);
                if (!sees) horizon.push_back({faces[f].V[k], faces[f].V[(k + 1) % 3], n, 0});
                if (visited[n] == search) continue;
                visited[n] = search;
                if (sees) stack.push_back(n);
            }
        }
        for (auto &edge : horizon) edge.NeighborEdge = edge_slot(faces[edge.Neighbor], edge.To, edge.From);
        // Each horizon edge fans to the point, and consecutive fans meet along the point's spokes.
        const uint32_t first_new = uint32_t(faces.size());
        std::unordered_map<uint32_t, uint32_t> fan_from, fan_to;
        for (uint32_t i = 0; i < horizon.size(); ++i) {
            const auto &edge = horizon[i];
            faces.push_back(make({edge.From, edge.To, p}));
            fan_from[edge.From] = first_new + i;
            fan_to[edge.To] = first_new + i;
        }
        // A horizon that is not one simple loop means the tolerance split a nearly coplanar region, and the hull is abandoned.
        if (fan_from.size() != horizon.size() || fan_to.size() != horizon.size()) return {};
        visited.resize(faces.size(), 0u);
        for (uint32_t i = 0; i < horizon.size(); ++i) {
            const auto &edge = horizon[i];
            auto &face = faces[first_new + i];
            face.Across = {edge.Neighbor, fan_from.at(edge.To), fan_to.at(edge.From)};
            faces[edge.Neighbor].Across[edge.NeighborEdge] = first_new + i;
        }
        // The visible faces' outside points move to the new faces.
        for (const auto f : visible) {
            auto &face = faces[f];
            face.Alive = false;
            for (const auto v : face.Outside)
                if (v != p) assign(v, first_new);
            face.Outside.clear();
        }
        enqueue(first_new);
    }
    std::vector<std::vector<uint32_t>> hull;
    for (const auto &face : faces)
        if (face.Alive) hull.push_back({face.V[0], face.V[1], face.V[2]});
    if (hull.empty()) return {};
    return PrimitiveListTask(meshes, mesh, hull);
}

std::optional<MeshTopologyTask> RotateEdgesTask(const Mesh &mesh, const GeometrySelection &selection) {
    mesh.ValidateSelection(selection);
    const auto &c = mesh.GetConnectivity();
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::RotateEdges, .Flags = TopologyFlagListSelects, .List = {0}, .SelectionElement = Element::Edge, .Selection = selection};
    std::ranges::for_each(selection.Edges, [&](uint32_t edge) {
        const auto h = mesh.GetHalfedge(he::EH{edge}, 0);
        const auto opposite = c.Opposites[*h];
        if (!opposite) return;
        task.List.push_back(*mesh.GetToVertex(c.Next(h)));
        task.List.push_back(*mesh.GetToVertex(c.Next(opposite)));
        task.List[0] += 2;
    });
    if (task.List[0] == 0) return {};
    return task;
}

std::optional<MeshTopologyTask> FillTask(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) {
    mesh.ValidateSelection(selection);
    const auto &vertices = selection.Vertices;
    if (vertices.size() < 2u) return {};
    const auto sources = SelectedEdgeSources(mesh, selection.Edges, false);
    const auto &c = mesh.GetConnectivity();
    if (vertices.size() == 2u) {
        if (sources.contains(MeshEdgeUsers::Key(vertices[0], vertices[1]))) return {};
        return PrimitiveListTask(meshes, mesh, std::array{vertices});
    }
    const auto selected_faces = selection.Faces.size();
    if (selected_faces) {
        if (selected_faces == 1u) return {};
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DissolveFaces, .SelectionElement = Element::Face, .Selection = selection};
    }
    auto loops = BoundaryLoops(meshes, mesh, selection.Edges, true);
    if (loops.empty()) {
        EdgeGraph neighbors;
        for (const auto &[key, h] : sources) {
            const auto a = uint32_t(key >> 32u), b = uint32_t(key);
            neighbors[a].push_back(b);
            neighbors[b].push_back(a);
        }
        if (std::ranges::all_of(neighbors, [](const auto &entry) { return entry.second.size() <= 2u; })) {
            std::vector<std::vector<uint32_t>> chains;
            VisitEdgeChains(neighbors, [&](const auto &chain, bool) { chains.push_back(chain); });
            // Complete an open chain through the single free selected point.
            if (chains.size() == 1u && neighbors.size() + 1u == vertices.size()) {
                for (const auto v : vertices)
                    if (!neighbors.contains(v)) chains[0].push_back(v);
            }
            for (auto &chain : chains)
                if (chain.size() >= 3u) loops.push_back(std::move(chain));
        }
        if (loops.empty()) {
            // Blender's vertex-cloud fallback orders points radially in their plane.
            // Read canonical positions directly; only handles and angular keys are stored on the host.
            vec3 center{};
            for (const auto v : vertices) center += mesh.GetPosition(he::VH{v}) / float(vertices.size());
            vec3 tangent{};
            for (const auto v : vertices) {
                const auto delta = mesh.GetPosition(he::VH{v}) - center;
                if (Dot(delta, delta) > Dot(tangent, tangent)) tangent = delta;
            }
            if (Dot(tangent, tangent) == 0.f) return {};
            tangent = Normalize(tangent);
            vec3 across{};
            for (const auto v : vertices) {
                auto delta = mesh.GetPosition(he::VH{v}) - center;
                delta -= tangent * Dot(delta, tangent);
                if (Dot(delta, delta) > Dot(across, across)) across = delta;
            }
            if (Dot(across, across) < 1e-20f) return {};
            across = Normalize(across);
            std::vector<std::pair<float, uint32_t>> angles;
            for (const auto v : vertices) {
                const auto delta = mesh.GetPosition(he::VH{v}) - center;
                angles.emplace_back(std::atan2(Dot(delta, across), Dot(delta, tangent)), v);
            }
            std::ranges::sort(angles);
            auto &loop = loops.emplace_back();
            for (const auto &[angle, v] : angles) loop.push_back(v);
        }
    }
    std::erase_if(loops, [&](const auto &loop) {
        const std::unordered_set<uint32_t> members(loop.begin(), loop.end());
        const auto fan = c.VertexCorners[loop.front()];
        for (uint32_t i = 0u; i < fan.y; ++i) {
            const auto face = c.FaceOf(he::HH{c.FanItems[fan.x + i].x});
            if (face && mesh.GetValence(face) == loop.size() &&
                std::ranges::all_of(mesh.fv_range(face), [&](auto v) { return members.contains(*v); })) return true;
        }
        return false;
    });
    if (loops.empty()) return {};
    for (auto &loop : loops) {
        int winding = 0;
        for (uint32_t i = 0u; i < loop.size(); ++i) {
            const auto a = loop[(i + loop.size() - 1u) % loop.size()], b = loop[i];
            const auto found = sources.find(MeshEdgeUsers::Key(a, b));
            if (found != sources.end() && c.FaceOf(he::HH{found->second}))
                winding += *mesh.GetFromVertex(he::HH{found->second}) == a ? 1 : -1;
        }
        if (winding > 0) std::ranges::reverse(loop);
    }
    return PrimitiveListTask(meshes, mesh, loops, sources);
}

MeshTopologyTask LoopCutTask(const Mesh &mesh, uint32_t edge, uint32_t cuts) {
    mesh.ValidateSelection({.Edges = {edge}});
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = float(std::max(cuts, 1u)), .Flags = TopologyFlagLoopCutSelect | TopologyFlagListSelects, .List = EdgeRing(mesh, edge)};
    task.List.insert(task.List.begin(), uint32_t(task.List.size()));
    task.Selection.Edges.assign(task.List.begin() + 1u, task.List.end());
    std::ranges::sort(task.Selection.Edges);
    return task;
}

MeshTopologyTask ExtrudeStepsTask(const Mesh &mesh, const GeometrySelection &selection, Element element, uint32_t steps, const mat3 &rotation, vec3 translation, vec3 center) {
    mesh.ValidateSelection(selection);
    if (element != Element::None && element != Element::Vertex && element != Element::Edge && element != Element::Face) throw std::invalid_argument("Invalid extrusion domain.");
    const auto displacement = center - rotation * center + translation;
    if (!FiniteMatrix(rotation, 3u) || !Finite(center) || !Finite(translation) || !Finite(displacement)) throw std::invalid_argument("Extrusion transform must be finite.");
    return {.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::ExtrudeRegion, .Flags = TopologyFlagTransformCopies, .Steps = std::max(steps, 1u), .CopyRotation = rotation, .CopyTranslation = displacement, .SelectionElement = element, .Selection = selection};
}

std::vector<MeshTopologyTask> BisectTasks(const Mesh &mesh, vec3 point, vec3 normal, bool clear_inner, bool clear_outer) {
    mesh.ValidateSelection({});
    const auto length = std::hypot(normal.x, normal.y, normal.z);
    if (!Finite(point) || !Finite(normal) || !(length > 0.f) || !std::isfinite(length)) throw std::invalid_argument("Bisect needs a finite plane with a nonzero normal.");
    const auto n = normal / length;
    if (!std::isfinite(Dot(n, point))) throw std::invalid_argument("Bisect plane offset exceeds its addressable range.");
    std::vector<MeshTopologyTask> tasks{{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagPlaneCuts | TopologyFlagLoopCutSelect | TopologyFlagSelectAll, .PlaneNormal = n, .PlaneOffset = Dot(n, point)}};
    for (const bool inner : {true, false}) {
        if (inner ? !clear_inner : !clear_outer) continue;
        const auto side = inner ? n : -n;
        tasks.push_back({.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DeleteFaces, .Flags = TopologyFlagPlaneSide | TopologyFlagSelectAll, .PlaneNormal = side, .PlaneOffset = Dot(side, point)});
    }
    return tasks;
}

std::vector<MeshTopologyTask> SymmetrizeTasks(const Mesh &mesh, uint8_t axis, bool negative) {
    if (axis > 2u) throw std::invalid_argument("Symmetry axis must be X, Y or Z.");
    vec3 normal{0.f};
    normal[axis] = negative ? -1.f : 1.f;
    auto tasks = BisectTasks(mesh, vec3{0.f}, normal, true, false);
    mat3 mirror{};
    for (int i = 0; i < 3; ++i) mirror[i][i] = i == axis ? -1.f : 1.f;
    tasks.push_back({.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies | TopologyFlagFlipCopies | TopologyFlagSelectAll, .CopyRotation = mirror});
    tasks.push_back({.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::MergeByDistance, .Param0 = 1e-5f, .Flags = TopologyFlagSelectAll});
    return tasks;
}

MeshTopologyTask KnifeTask(const Mesh &mesh, const GeometrySelection &selection, const mat4 &mesh_to_clip, vec2 extent, vec2 start, vec2 end) {
    mesh.ValidateSelection(selection);
    if (!FiniteMatrix(mesh_to_clip, 4u) || !Finite(extent) || !(extent.x > 0.f && extent.y > 0.f) || !Finite(start) || !Finite(end)) throw std::invalid_argument("Knife needs a finite transform and positive screen extent.");
    return {.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagScreenCuts | TopologyFlagLoopCutSelect, .ScreenTransform = mesh_to_clip, .Extent = extent, .KnifeStart = start, .KnifeEnd = end, .Selection = selection};
}

std::optional<MeshTopologyTask> MergeTask(const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection, GeometryMergeMode mode, float distance, uint32_t reference) {
    mesh.ValidateSelection(selection);
    using Mode = GeometryMergeMode;
    if (uint32_t(mode) > uint32_t(Mode::ByDistance) || !std::isfinite(distance)) throw std::invalid_argument("Invalid merge mode or distance.");
    if (mode == Mode::Collapse || mode == Mode::ByDistance)
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = mode == Mode::Collapse ? MeshTopologyOp::MergeCollapse : MeshTopologyOp::MergeByDistance, .Param0 = std::max(distance, 0.f), .SelectionElement = Element::Vertex, .Selection = selection};
    if (selection.Vertices.size() < 2u) return {};
    if (!meshes.IsLiveElement(mesh.GetStoreId(), Element::Vertex, reference) || !std::ranges::binary_search(selection.Vertices, reference)) throw std::invalid_argument("Merge reference must belong to the selection.");
    const auto target = reference;
    vec3 position = mesh.GetPosition(he::VH{target});
    if (mode == Mode::Center) {
        position = {};
        for (const auto vertex : selection.Vertices) position += mesh.GetPosition(he::VH{vertex});
        position /= float(selection.Vertices.size());
    }
    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::MergeAtTarget, .TargetVertex = target, .TargetPosition = position, .SelectionElement = Element::Vertex, .Selection = selection};
}
