#include "mesh/TopologyOperations.h"
#include "SortUnique.h"
#include "gpu/ConvexHull.h"
#include "gpu/PartitionDivisions.h"
#include "mesh/AttributeFields.h"
#include "mesh/BoundaryLoops.h"
#include "mesh/ConvexHull.h"
#include "mesh/EdgeKey.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshStore.h"
#include "mesh/Neighborhood.h"
#include "mesh/Partition.h"
#include "mesh/SelectedChains.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "state/Registry.h"
#include <map>
#include <set>

namespace {
bool FiniteMatrix(const auto &value, uint32_t size) {
    for (uint32_t column = 0u; column < size; ++column) {
        for (uint32_t row = 0u; row < size; ++row) {
            if (!std::isfinite(value[column][row])) return false;
        }
    }
    return true;
}
std::vector<CornerLoop> WithSources(std::span<const std::vector<uint32_t>> loops, const std::unordered_map<uint64_t, uint32_t> &sources) {
    std::vector<CornerLoop> corners;
    for (const auto &loop : loops) {
        auto &pairs = corners.emplace_back();
        for (uint32_t i = 0u; i < loop.size(); ++i) {
            const auto edge = sources.find(EdgeKey(loop[(i + loop.size() - 1u) % loop.size()], loop[i]));
            pairs.push_back({loop[i], edge == sources.end() ? InvalidOffset : edge->second});
        }
    }
    return corners;
}
// A loop of two becomes a loose edge.
MeshTopologyTask PrimitiveListTask(const MeshStore &meshes, const Mesh &mesh, std::span<const CornerLoop> loops, std::span<const uint32_t> grid_loop = {}, uint32_t grid_span = 0u) {
    const auto appended_base = meshes.Arenas().Vertices.HandleBound(meshes.Get(mesh.GetStoreId()).Vertices);
    const uint64_t vertices = grid_loop.empty() ? 0u : uint64_t(grid_span - 1u) * (grid_loop.size() / 2u - grid_span - 1u);
    if (uint64_t(appended_base) + vertices > UINT32_MAX) throw std::length_error("Face list exceeds the vertex handle address space.");
    MeshTopologyTask task{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::AddPrimitives, .AppendedBase = appended_base};
    uint32_t attribute_source = InvalidOffset;
    for (const auto &loop : loops) {
        for (const auto &[vertex, source] : loop) {
            if (vertex < appended_base && attribute_source == InvalidOffset) attribute_source = vertex;
        }
    }
    if (attribute_source == InvalidOffset) throw std::invalid_argument("Face creation needs a source vertex for attributes.");
    task.List = {uint32_t(vertices), uint32_t(grid_loop.size()), grid_span, attribute_source};
    task.List.insert(task.List.end(), grid_loop.begin(), grid_loop.end());
    task.List.push_back(uint32_t(loops.size()));
    for (const auto &loop : loops) {
        task.List.push_back(uint32_t(loop.size()));
        for (const auto &[vertex, source] : loop) {
            task.List.push_back(vertex);
            task.List.push_back(source);
        }
    }
    return task;
}

std::unordered_map<uint64_t, uint32_t> SourcesOf(const SelectedChains &chains) {
    std::unordered_map<uint64_t, uint32_t> sources;
    for (const auto &edge : chains.Edges) {
        if (edge.From != edge.To && edge.Source != InvalidOffset) sources.try_emplace(EdgeKey(edge.From, edge.To), edge.Source);
    }
    return sources;
}
// The loop reversed, each vertex keeping the source of the edge now before it.
CornerLoop Reversed(const CornerLoop &loop) {
    const auto n = loop.size();
    CornerLoop reversed(n);
    for (size_t j = 0u; j < n; ++j) reversed[j] = {loop[n - 1u - j][0], loop[(n - j) % n][1]};
    return reversed;
}
} // namespace

std::vector<MeshTopologyTask> SeparateGeometryTasks(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection, GeometrySeparateMode mode) {
    const auto &meshes = r.Resources.get<const MeshStore>();
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
        // Each component's label is its lowest vertex, so the groups come in order of their first vertices.
        const auto vertices = HandleList(mesh.vertices());
        const auto labels = ComponentLabels(meshes, GetMeshPipelines(r), id, Element::Vertex, vertices, FieldAdjacency::Domain);
        std::map<uint32_t, std::vector<uint32_t>> components;
        for (uint32_t i = 0u; i < vertices.size(); ++i) components[labels[i]].push_back(vertices[i]);
        for (auto &[label, group] : components) groups.push_back(std::move(group));
        // Blender keeps the first connected component in the original object.
        if (!groups.empty()) groups.erase(groups.begin());
    } else {
        element = Element::Face;
        const auto palette = meshes.Arenas().PrimitiveMaterials.Get(meshes.Get(id).PrimitiveMaterials);
        const auto faces = HandleList(mesh.faces());
        const auto primitives = FacePrimitives(meshes, GetMeshPipelines(r), id, faces);
        std::unordered_map<uint32_t, uint32_t> materials;
        for (uint32_t f = 0u; f < faces.size(); ++f) {
            const auto material = palette[primitives[f]];
            const auto [entry, inserted] = materials.try_emplace(material, uint32_t(groups.size()));
            if (inserted) groups.emplace_back();
            groups[entry->second].push_back(faces[f]);
        }
        // Blender extracts successive material groups, leaving the last in place.
        if (!groups.empty()) groups.pop_back();
    }
    if (groups.empty()) return {};
    const auto group_selection = [&](std::vector<uint32_t> handles) {
        return element == Element::Face ? GeometrySelection{.Faces = std::move(handles)} : GeometrySelection{.Vertices = std::move(handles)};
    };
    std::vector<uint32_t> removed;
    for (auto &group : groups) {
        std::ranges::sort(group);
        removed.insert(removed.end(), group.begin(), group.end());
        tasks.push_back({.SourceId = id, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = element, .Selection = group_selection(std::move(group))});
    }
    std::ranges::sort(removed);
    tasks.push_back({.SourceId = id, .Op = element == Element::Face ? MeshTopologyOp::DeleteFaces : mesh.FaceCount() ? MeshTopologyOp::DeleteVertices :
                                                                                                                       MeshTopologyOp::DeleteEdges,
                     .SelectionElement = element,
                     .Selection = group_selection(std::move(removed))});
    return tasks;
}

void RecordBridgeEdgeLoopsTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, std::function<void(std::optional<MeshTopologyTask>)> then) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto id = mesh.GetStoreId();
    RecordSelectedEdgeChains(meshes, pipelines, chain, id, selection.Edges, true, [&meshes, &pipelines, &chain, id, then = std::move(then)](SelectedChains chains) mutable {
        if (chains.Junction || chains.Chains.size() != 2u || chains.Chains[0].Closed != chains.Chains[1].Closed) return then({});
        if (chains.Chains[0].Pairs.size() < chains.Chains[1].Pairs.size()) std::swap(chains.Chains[0], chains.Chains[1]);
        const std::array loops{chains.Chains[0].Pairs, chains.Chains[1].Pairs};
        RecordSummarizeLoops(meshes, pipelines, chain, id, loops, [&meshes, &pipelines, &chain, id, chains = std::move(chains), then = std::move(then)](std::vector<ChainSummary> summaries) mutable {
            const bool closed = chains.Chains[0].Closed;
            auto a = VerticesOf(chains.Chains[0].Pairs), b = VerticesOf(chains.Chains[1].Pairs);
            const auto flip = [&](std::vector<uint32_t> &chain, ChainSummary &summary) {
                std::ranges::reverse(chain | std::views::drop(closed ? 1u : 0u));
                summary.Normal = summary.Normal * -1.f;
                if (!closed) std::swap(summary.First, summary.Last);
            };
            // New faces oppose the existing face along each rail.
            if (summaries[0].Winding < 0) flip(a, summaries[0]);
            if (summaries[1].Winding > 0) flip(b, summaries[1]);
            const auto align = [&](const ChainSummary &fixed, std::vector<uint32_t> &free_chain, ChainSummary &free) {
                if (closed) {
                    if (Dot(fixed.Normal, free.Normal) < 0.f) flip(free_chain, free);
                } else {
                    const auto same = Distance2(fixed.First, free.First) + Distance2(fixed.Last, free.Last);
                    const auto crossed = Distance2(fixed.First, free.Last) + Distance2(fixed.Last, free.First);
                    if (crossed < same) flip(free_chain, free);
                }
            };
            if (!summaries[1].Winding) align(summaries[0], b, summaries[1]);
            else if (!summaries[0].Winding) align(summaries[1], a, summaries[0]);
            // Each rail pairs its vertices with the other's, the second starting nearest the first's first vertex when the rails close.
            const auto bridge = [&meshes, id, closed, chains = std::move(chains), then = std::move(then)](const std::vector<uint32_t> &a, const std::vector<uint32_t> &b) {
                const auto na = uint32_t(a.size()) - uint32_t(!closed), nb = uint32_t(b.size()) - uint32_t(!closed);
                std::vector<std::vector<uint32_t>> faces;
                faces.reserve(na);
                for (uint32_t i = 0u; i < na; ++i) {
                    const auto j0 = uint32_t(uint64_t(i) * nb / na), j1 = uint32_t(uint64_t(i + 1u) * nb / na);
                    const auto next = a[(i + 1u) % a.size()];
                    if (j0 == j1) faces.push_back({next, a[i], b[j0]});
                    else faces.push_back({next, a[i], b[j0], b[j1 % b.size()]});
                }
                then(PrimitiveListTask(meshes, Mesh{meshes, id}, WithSources(faces, SourcesOf(chains))));
            };
            if (!closed) return bridge(a, b);
            if (!summaries[0].Winding && !summaries[1].Winding) {
                if (Dot(summaries[0].Normal, summaries[0].Centroid - summaries[1].Centroid) < 0.f) {
                    flip(a, summaries[0]);
                    flip(b, summaries[1]);
                }
            }
            CornerLoop current(b.size());
            for (uint32_t j = 0u; j < b.size(); ++j) current[j] = {b[j], InvalidOffset};
            RecordNearestInLoop(meshes, pipelines, chain, id, current, summaries[0].First, [a = std::move(a), b = std::move(b), bridge = std::move(bridge)](uint32_t nearest) mutable {
                std::ranges::rotate(b, b.begin() + nearest);
                bridge(a, b);
            });
        });
    });
}

void RecordGridFillTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, uint32_t span, std::function<void(std::optional<MeshTopologyTask>)> then) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto id = mesh.GetStoreId();
    RecordSelectedEdgeChains(meshes, pipelines, chain, id, selection.Edges, true, [&meshes, &pipelines, &chain, id, span, then = std::move(then)](SelectedChains chains) mutable {
        if (chains.Junction || chains.Chains.size() != 1u || !chains.Chains[0].Closed || chains.Chains[0].Pairs.size() % 2u || chains.Chains[0].Pairs.size() < 4u) return then({});
        RecordSummarizeLoops(meshes, pipelines, chain, id, std::span{&chains.Chains[0].Pairs, 1u}, [&meshes, id, span, chains = std::move(chains), then = std::move(then)](std::vector<ChainSummary> summaries) {
            auto loop = VerticesOf(chains.Chains[0].Pairs);
            if (summaries[0].Winding < 0) std::ranges::reverse(loop | std::views::drop(1));
            const auto length = uint32_t(loop.size());
            const auto appended_base = meshes.Arenas().Vertices.HandleBound(meshes.Get(id).Vertices);
            const uint32_t s = std::clamp(span == 0 ? std::max(length / 4, 1u) : span, 1u, length / 2 - 1), t = length / 2 - s;
            const uint64_t interior = uint64_t(s - 1u) * (t - 1u), cells = uint64_t(s) * t;
            if (uint64_t(appended_base) + interior > UINT32_MAX || 5ull + length + 9u * cells > UINT32_MAX) {
                throw std::length_error("Grid fill exceeds the vertex or face-list address space.");
            }
            // Nodes run along the first side and up the second, with the loop's four sides as the rails.
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
            then(PrimitiveListTask(meshes, Mesh{meshes, id}, WithSources(faces, SourcesOf(chains)), loop, s));
        });
    });
}

std::optional<MeshTopologyTask> SubdivideLengthTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection, float length) {
    if (!(length > 0.f)) return {};
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto id = mesh.GetStoreId();
    MeshTopologyTask task{.SourceId = id, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagListCounts, .SelectionElement = Element::Edge, .Selection = selection};
    auto &list = task.List;
    list.push_back(0u);
    std::vector<uint32_t> cuts;
    uint32_t most = 0u;
    const auto ends = SelectedEdgeFacts(meshes, pipelines, id, selection.Edges, {});
    std::vector<uint32_t> vertices;
    for (const auto &edge : ends) vertices.insert(vertices.end(), {edge.From, edge.To});
    SortUnique(vertices);
    const auto positions = VertexPositions(meshes, id, vertices);
    const auto position = [&](uint32_t v) { return positions[uint32_t(std::ranges::lower_bound(vertices, v) - vertices.begin())]; };
    for (uint32_t i = 0u; i < selection.Edges.size(); ++i) {
        const auto edge = selection.Edges[i];
        // Manifold measures in double, which decides ratios that land near a whole number.
        const auto a = position(ends[i].To), b = position(ends[i].From);
        const double dx = double(a.x) - double(b.x), dy = double(a.y) - double(b.y), dz = double(a.z) - double(b.z);
        const auto count = uint32_t(std::sqrt(dx * dx + dy * dy + dz * dz) / double(length));
        if (edge >= cuts.size()) cuts.resize(edge + 1u, 0u);
        cuts[edge] = count;
        if (!count) continue;
        list.insert(list.end(), {edge, count});
        ++list[0];
        most = std::max(most, count);
    }
    if (!most) return {};
    task.Param0 = float(most);
    std::set<std::array<uint32_t, 4>> tables;
    const auto faces = HandleList(mesh.faces());
    const auto corners = FaceCorners(meshes, pipelines, id, faces);
    for (uint32_t f = 0u; f < faces.size(); ++f) {
        std::array<uint32_t, 4> divisions{};
        uint32_t sides = 0u;
        bool cut = true;
        for (const auto &corner : corners.Of(f)) {
            const auto edge = corner.Edge;
            const auto count = edge < cuts.size() ? cuts[edge] : 0u;
            cut = cut && count > 0u;
            if (sides < 4u) divisions[sides] = count + 1u;
            ++sides;
        }
        if (!cut || sides < 3u || sides > 4u || std::ranges::all_of(divisions | std::views::take(sides), [&](uint32_t d) { return d == divisions[0]; })) continue;
        // The kernel's side c runs from corner c along the edge arriving at the next corner.
        // A quad's sorted divisions depend on its first side, so the table's key starts where the kernel's does.
        std::rotate(divisions.begin(), divisions.begin() + 1, divisions.begin() + sides);
        tables.insert(SortDivisions(divisions).Sorted);
    }
    list.push_back(uint32_t(tables.size()));
    for (const auto &sorted : tables) {
        const auto table = Partition(sorted);
        list.insert(list.end(), sorted.begin(), sorted.end());
        list.insert(list.end(), {uint32_t(table.Weights.size()), uint32_t(table.Weights.size()) - table.InteriorOffset(), uint32_t(table.Triangles.size())});
        for (const auto &w : table.Weights) list.insert(list.end(), {std::bit_cast<uint32_t>(w.x), std::bit_cast<uint32_t>(w.y), std::bit_cast<uint32_t>(w.z), std::bit_cast<uint32_t>(w.w)});
        for (const auto &t : table.Triangles) list.insert(list.end(), t.begin(), t.end());
    }
    return task;
}

std::vector<MeshTopologyTask> SmoothSurfaceTasks(const Mesh &mesh, uint32_t cuts) {
    const auto &c = mesh.GetConnectivity();
    GeometrySelection polygons;
    for (const auto face : mesh.faces()) {
        const auto corners = c.FaceRange(*face);
        if (corners.y - corners.x > 4u) polygons.Faces.push_back(*face);
    }
    std::vector<MeshTopologyTask> tasks;
    if (!polygons.Faces.empty()) tasks.push_back({.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = std::move(polygons)});
    tasks.push_back({.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = float(std::max(cuts, 1u)), .Flags = TopologyFlagSelectAll | TopologyFlagBezier});
    return tasks;
}

namespace {
// A record's convex hull request, which the gathered requests' completion takes.
struct HullRequest {
    uint32_t StoreId;
    GeometrySelection Selection;
    std::function<void(std::vector<MeshTopologyTask>)> Then;
};
// The tasks that replace a record's selection with the hull of its selected vertices, whose loops index those vertices.
std::vector<MeshTopologyTask> HullTasks(const MeshStore &meshes, uint32_t id, const GeometrySelection &selection, std::vector<std::vector<uint32_t>> hull) {
    if (hull.empty()) return {};
    const auto &points = selection.Vertices;
    for (auto &loop : hull) {
        for (auto &v : loop) v = points[v];
    }
    const Mesh mesh{meshes, id};
    // A planar selection's polygon faces the way the mesh's faces do.
    if (hull.size() == 1u && mesh.FaceCount()) {
        const auto positions = VertexPositions(meshes, id, hull[0]);
        vec3 normal{};
        for (uint32_t i = 1u; i + 1u < hull[0].size(); ++i) normal += Cross(positions[i] - positions[0], positions[i + 1u] - positions[0]);
        const std::array face{**mesh.faces().begin()};
        if (Dot(normal, FaceNormals(meshes, id, face).front()) < 0.f) std::ranges::reverse(hull[0]);
    }
    std::vector<MeshTopologyTask> tasks;
    if (!selection.Faces.empty()) tasks.push_back({.SourceId = id, .Op = MeshTopologyOp::DeleteOnlyFaces, .SelectionElement = Element::Face, .Selection = GeometrySelection{.Faces = selection.Faces}});
    const GeometrySelection remaining{.Vertices = selection.Vertices, .Edges = selection.Edges};
    auto faces = PrimitiveListTask(meshes, mesh, WithSources(hull, {}));
    faces.Selection = remaining;
    tasks.push_back(std::move(faces));
    tasks.push_back({.SourceId = id, .Op = MeshTopologyOp::DeleteLoose, .Selection = remaining});
    return tasks;
}
// Computes the hulls of the requests' selected vertices in one quickhull on a chain of its own and calls each request's `Then` with its tasks.
void RunHulls(state::Registry &r, std::span<const HullRequest> requests) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<uint32_t> first_point{0u};
    for (const auto &request : requests) first_point.push_back(first_point.back() + uint32_t(request.Selection.Vertices.size()));
    const auto sets = uint32_t(requests.size());
    const auto positions = chain.Scratch.Allocate(3u * (first_point.back() + sets));
    std::vector<HullPointsJob> jobs;
    for (uint32_t s = 0u; s < sets; ++s) {
        const auto &points = requests[s].Selection.Vertices;
        jobs.push_back({
            .Positions = {chain.Scratch.Buffer.Slot, positions.Offset + 3u * first_point[s]},
            .Count = uint32_t(points.size()),
            .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
            .First = meshes.GetConnectivityRef(requests[s].StoreId),
            .FirstHandles = chain.Upload(std::as_bytes(std::span{points})),
        });
    }
    mtl::GatherPass(chain, GetMeshPipelines(r)[MeshPass::HullGather], std::span{jobs}, &HullPointsJob::Count);
    auto hulls = ConvexHulls(r, chain, positions.Offset, first_point);
    for (uint32_t s = 0u; s < sets; ++s) requests[s].Then(HullTasks(meshes, requests[s].StoreId, requests[s].Selection, std::move(hulls[s])));
}
} // namespace

void RecordConvexHullTasks(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, std::function<void(std::vector<MeshTopologyTask>)> then) {
    if (selection.Vertices.size() < 4u) return then({});
    static constexpr char Hulls{};
    chain.Lane(mesh.GetStoreId(), [&] {
        chain.Gather(&Hulls, std::vector{HullRequest{mesh.GetStoreId(), selection, std::move(then)}}, 0u, [&r, &chain](std::vector<HullRequest> requests) {
            chain.AfterSubmit([&r, requests = std::move(requests)] { RunHulls(r, requests); });
        });
    });
}
std::vector<MeshTopologyTask> ConvexHullTasks(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
    return mtl::Drained<std::vector<MeshTopologyTask>>(r.Resources.get<const MeshStore>().BufferContext(), [&](mtl::ComputeChain &chain, auto then) { RecordConvexHullTasks(r, chain, mesh, selection, std::move(then)); });
}

void RecordFillTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, const GeometrySelection &selection, uint32_t sides, std::function<void(std::optional<MeshTopologyTask>)> then) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    const auto id = mesh.GetStoreId();
    const auto &vertices = selection.Vertices;
    if (vertices.size() < 2u) return then({});
    if (vertices.size() == 2u) {
        return RecordSelectedEdgeChains(meshes, pipelines, chain, id, selection.Edges, false, [&meshes, id, vertices, then = std::move(then)](SelectedChains chains) {
            if (std::ranges::any_of(chains.Chains, [](const SelectedChain &chain) { return chain.Pairs.size() == 2u; })) return then({});
            then(PrimitiveListTask(meshes, Mesh{meshes, id}, WithSources(std::array{vertices}, {})));
        });
    }
    if (const auto selected_faces = selection.Faces.size()) {
        if (selected_faces == 1u) return then({});
        return then(MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::DissolveFaces, .SelectionElement = Element::Face, .Selection = selection});
    }
    RecordBoundaryLoops(meshes, pipelines, chain, id, selection.Edges, sides, [&meshes, &pipelines, &chain, id, selection, then = std::move(then)](std::vector<CornerLoop> boundaries) mutable {
        if (!boundaries.empty()) return then(PrimitiveListTask(meshes, Mesh{meshes, id}, boundaries));
        RecordSelectedEdgeChains(meshes, pipelines, chain, id, selection.Edges, false, [&meshes, &pipelines, id, selection, then = std::move(then)](SelectedChains chains) {
            const auto &vertices = selection.Vertices;
            std::vector<CornerLoop> loops;
            if (!chains.Junction) {
                auto chained = chains.Chains;
                size_t covered = 0u;
                for (const auto &chain : chained) covered += chain.Pairs.size();
                if (chained.size() == 1u && covered + 1u == vertices.size()) {
                    auto &pairs = chained[0].Pairs;
                    for (const auto v : vertices) {
                        if (std::ranges::none_of(pairs, [&](const auto &pair) { return pair[0] == v; })) pairs.push_back({v, InvalidOffset});
                    }
                    pairs[0][1] = InvalidOffset;
                }
                for (auto &chain : chained) {
                    if (chain.Pairs.size() >= 3u) loops.push_back(std::move(chain.Pairs));
                }
            }
            if (loops.empty()) {
                // Blender's vertex-cloud fallback orders points radially in their plane.
                const auto angles = PointCloudAngles(meshes, pipelines, id, vertices);
                if (!angles) return then({});
                std::vector<std::pair<float, uint32_t>> order;
                for (uint32_t i = 0u; i < vertices.size(); ++i) order.emplace_back((*angles)[i], vertices[i]);
                std::ranges::sort(order);
                std::vector<uint32_t> loop;
                for (const auto &[angle, v] : order) loop.push_back(v);
                loops = WithSources(std::array{loop}, SourcesOf(chains));
            }
            const auto summaries = SummarizeLoops(meshes, pipelines, id, loops);
            // A loop a face already spans adds nothing, and one winding with its faces turns against them.
            std::vector<CornerLoop> filled;
            for (uint32_t i = 0u; i < loops.size(); ++i) {
                if (summaries[i].MatchesFace) continue;
                filled.push_back(summaries[i].Winding > 0 ? Reversed(loops[i]) : std::move(loops[i]));
            }
            if (filled.empty()) return then({});
            then(PrimitiveListTask(meshes, Mesh{meshes, id}, filled));
        });
    });
}

MeshTopologyTask ExtrudeStepsTask(const Mesh &mesh, const GeometrySelection &selection, Element element, uint32_t steps, const mat3 &rotation, vec3 translation, vec3 center) {
    const auto displacement = center - rotation * center + translation;
    if (!FiniteMatrix(rotation, 3u) || !Finite(center) || !Finite(translation) || !Finite(displacement)) throw std::invalid_argument("Extrusion transform must be finite.");
    return {.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::ExtrudeRegion, .Flags = TopologyFlagTransformCopies, .Steps = std::max(steps, 1u), .CopyRotation = rotation, .CopyTranslation = displacement, .SelectionElement = element, .Selection = selection};
}

MeshTopologyTask PlaneCutTask(const Mesh &mesh, vec3 point, vec3 normal) {
    const auto n = normal / std::hypot(normal.x, normal.y, normal.z);
    return {.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagPlaneCuts | TopologyFlagNewLoopsOnly | TopologyFlagSelectAll, .PlaneNormal = n, .PlaneOffset = Dot(n, point)};
}
void RecordPlaneSideDeleteTask(state::Registry &r, mtl::ComputeChain &chain, const Mesh &mesh, vec3 point, vec3 normal, PlaneSideKind side, float tolerance, std::function<void(std::optional<MeshTopologyTask>)> then) {
    auto &meshes = r.Resources.get<MeshStore>();
    const auto id = mesh.GetStoreId();
    const auto field = meshes.AddAttribute(id, AttributeDomain::Vertex, AttributeType::Bool, ".plane side").value();
    chain.Lane([&] { RecordPlaneSide(meshes, GetMeshPipelines(r), chain, field, point, normal, side, tolerance); });
    chain.AfterSubmit([&meshes, id, field, then = std::move(then)] {
        const auto selected = meshes.AttributeView<bool>(field);
        std::vector<uint32_t> vertices;
        for (const auto v : Mesh{meshes, id}.vertices()) {
            if (selected[*v]) vertices.push_back(*v);
        }
        meshes.ReleaseAttribute(field);
        if (vertices.empty()) return then({});
        then(MeshTopologyTask{.SourceId = id, .Op = MeshTopologyOp::DeleteVertices, .SelectionElement = Element::Vertex, .Selection = GeometrySelection{.Vertices = std::move(vertices)}});
    });
}

MeshTopologyTask KnifeTask(const Mesh &mesh, const mat4 &mesh_to_clip, vec2 extent, vec2 start, vec2 end) {
    if (!FiniteMatrix(mesh_to_clip, 4u) || !Finite(extent) || !(extent.x > 0.f && extent.y > 0.f) || !Finite(start) || !Finite(end)) throw std::invalid_argument("Knife needs a finite transform and positive screen extent.");
    return {.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::Subdivide, .Param0 = 1.f, .Flags = TopologyFlagScreenCuts | TopologyFlagNewLoopsOnly | TopologyFlagSelectAll, .ScreenTransform = mesh_to_clip, .Extent = extent, .KnifeStart = start, .KnifeEnd = end};
}

std::optional<MeshTopologyTask> MergeTask(state::Registry &r, const Mesh &mesh, const GeometrySelection &selection, GeometryMergeMode mode, float distance, uint32_t reference) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    using Mode = GeometryMergeMode;
    if (mode == Mode::Collapse || mode == Mode::ByDistance) {
        return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = mode == Mode::Collapse ? MeshTopologyOp::MergeCollapse : MeshTopologyOp::MergeByDistance, .Param0 = std::max(distance, 0.f), .SelectionElement = Element::Vertex, .Selection = selection};
    }
    if (selection.Vertices.size() < 2u) return {};
    const auto position = [&] {
        if (mode != Mode::Center) return VertexPositions(meshes, mesh.GetStoreId(), std::array{reference}).front();
        vec3 center{};
        for (const auto &p : VertexPositions(meshes, mesh.GetStoreId(), selection.Vertices)) center += p;
        center /= float(selection.Vertices.size());
        return center;
    }();
    return MeshTopologyTask{.SourceId = mesh.GetStoreId(), .Op = MeshTopologyOp::MergeAtTarget, .TargetVertex = reference, .TargetPosition = position, .SelectionElement = Element::Vertex, .Selection = selection};
}

std::optional<MeshTopologyTask> BooleanTask(const Mesh &first, const Mesh &second, BooleanOperation operation, const mat3 &rotation, vec3 translation) {
    if (!first.HasClosedSurface() || !second.HasClosedSurface()) return {};
    return MeshTopologyTask{
        .SourceId = first.GetStoreId(),
        .Op = MeshTopologyOp::Boolean,
        .SecondSourceId = second.GetStoreId(),
        .Param0 = float(uint32_t(operation)),
        .Flags = TopologyFlagSelectAll,
        .CopyRotation = rotation,
        .CopyTranslation = translation,
    };
}
