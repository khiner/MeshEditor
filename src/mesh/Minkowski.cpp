#include "mesh/Minkowski.h"
#include "gpu/ConvexHull.h"
#include "mesh/ConvexHull.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/MeshCreate.h"
#include "mesh/Neighborhood.h"
#include "mesh/Region.h"
#include "mesh/ScratchPasses.h"
#include "mesh/TopologyOperations.h"
#include "state/Registry.h"

#include <numbers>
#include <numeric>

namespace {
struct Geometry : FaceLoops {
    std::vector<vec3> Normals;

    bool Convex(const Mesh &mesh) const {
        if (int64_t(mesh.VertexCount()) - int64_t(mesh.EdgeCount()) + int64_t(mesh.FaceCount()) != 2) return false;
        float extent = 0.f;
        for (const auto v : mesh.vertices()) extent = std::max(extent, Length(Positions[*v]));
        const float tolerance = 1e-5f * std::max(1.f, extent);
        for (uint32_t f = 0u; f < Faces.size(); ++f) {
            const auto origin = Positions[Corners.Of(f).front().Vertex];
            for (const auto v : mesh.vertices()) {
                if (Dot(Normals[f], Positions[*v] - origin) > tolerance) return false;
            }
        }
        return true;
    }
    bool ConvexRegion(const RegionPlane &plane) const {
        if (Faces.size() != 1u) return false;
        const auto corners = Corners.Of(0u);
        for (size_t i = 0; i < corners.size(); ++i) {
            const auto a = Positions[corners[i].Vertex], b = Positions[corners[(i + 1) % corners.size()].Vertex], c = Positions[corners[(i + 2) % corners.size()].Vertex];
            if (Dot(Cross(b - a, c - b), plane.Normal) < 0.f) return false;
        }
        return true;
    }
    std::vector<uint32_t> EdgeHandles() const {
        std::vector<uint32_t> handles;
        for (uint32_t f = 0u; f < Faces.size(); ++f) {
            const auto corners = Corners.Of(f);
            for (size_t i = 0; i < corners.size(); ++i) handles.insert(handles.end(), {corners[i].Vertex, corners[(i + 1) % corners.size()].Vertex});
        }
        return handles;
    }
    vec3 TriangleNormal(std::span<const uint32_t> handles) const {
        const auto a = Positions[handles[0]];
        return Normalize(Cross(Positions[handles[1]] - a, Positions[handles[2]] - a));
    }
    vec3 Centroid(const Mesh &mesh) const {
        vec3 sum{};
        for (const auto v : mesh.vertices()) sum += Positions[*v];
        return mesh.VertexCount() ? sum / float(mesh.VertexCount()) : sum;
    }
};
// The connected components of the face loops, each with one of its vertices and its bounds, and each face's component.
struct Components {
    struct Component {
        uint32_t Vertex;
        vec3 Min, Max;
    };
    std::vector<Component> Of;
    std::vector<uint32_t> FaceComponent;
};
Components ConnectedComponents(const Geometry &geometry) {
    std::vector<uint32_t> parent(geometry.Positions.size());
    std::iota(parent.begin(), parent.end(), 0u);
    const auto root = [&](uint32_t v) {
        while (parent[v] != v) v = parent[v] = parent[parent[v]];
        return v;
    };
    for (uint32_t f = 0u; f < geometry.Faces.size(); ++f) {
        const auto corners = geometry.Corners.Of(f);
        for (const auto corner : corners) parent[root(corner.Vertex)] = root(corners.front().Vertex);
    }
    Components components;
    std::vector<uint32_t> index(geometry.Positions.size(), InvalidOffset);
    for (uint32_t f = 0u; f < geometry.Faces.size(); ++f) {
        const auto corners = geometry.Corners.Of(f);
        const auto r = root(corners.front().Vertex);
        if (index[r] == InvalidOffset) {
            index[r] = uint32_t(components.Of.size());
            components.Of.push_back({r, geometry.Positions[r], geometry.Positions[r]});
        }
        auto &component = components.Of[index[r]];
        for (const auto corner : corners) {
            component.Min = Min(component.Min, geometry.Positions[corner.Vertex]);
            component.Max = Max(component.Max, geometry.Positions[corner.Vertex]);
        }
        components.FaceComponent.push_back(index[r]);
    }
    return components;
}
// Whether each component's face loops wind around the point, by the solid angles of a solid's fan triangles or the angles of a region's edges about the plane normal, summed in double.
std::vector<bool> Winds(const Geometry &geometry, const Components &components, vec3 point, const std::optional<RegionPlane> &plane) {
    using numeric::dvec3;
    const auto at = [&](uint32_t v) { return dvec3{geometry.Positions[v]} - dvec3{point}; };
    std::vector<double> turns(components.Of.size(), 0.0);
    for (uint32_t f = 0u; f < components.FaceComponent.size(); ++f) {
        const auto corners = geometry.Corners.Of(f);
        auto &turn = turns[components.FaceComponent[f]];
        for (size_t i = 0u; i < corners.size(); ++i) {
            const auto a = at(corners[i].Vertex), b = at(corners[(i + 1u) % corners.size()].Vertex);
            if (plane) {
                turn += std::atan2(Dot(Cross(a, b), dvec3{plane->Normal}), Dot(a, b)) / (2.0 * std::numbers::pi);
            } else if (i > 0u && i + 1u < corners.size()) {
                const auto o = at(corners.front().Vertex);
                const double la = Length(o), lb = Length(a), lc = Length(b);
                turn += std::atan2(Dot(o, Cross(a, b)), la * lb * lc + Dot(o, a) * lc + Dot(o, b) * lb + Dot(a, b) * la) / (2.0 * std::numbers::pi);
            }
        }
    }
    return turns | std::views::transform([](double t) { return std::abs(t) >= .5; }) | std::ranges::to<std::vector>();
}
// The brush reflected through the origin, with a solid's loops reversed so its faces still face out.
// The reflection of a region turns it within its plane, which keeps its loops' direction.
MeshSource Reflection(const Geometry &brush, bool solid) {
    MeshData data;
    std::vector<uint32_t> index(brush.Positions.size(), InvalidOffset), loop;
    for (uint32_t f = 0u; f < brush.Faces.size(); ++f) {
        loop.clear();
        for (const auto corner : brush.Corners.Of(f)) {
            if (index[corner.Vertex] == InvalidOffset) {
                index[corner.Vertex] = uint32_t(data.Positions.size());
                data.Positions.push_back(brush.Positions[corner.Vertex] * -1.f);
            }
            loop.push_back(index[corner.Vertex]);
        }
        if (solid) std::ranges::reverse(loop);
        data.AddFace(loop);
    }
    return {.Data = std::move(data)};
}
// Each record's face loops, positions and face normals, in one gather of each.
std::vector<Geometry> ReadGeometry(state::Registry &r, std::span<const uint32_t> ids) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    std::vector<Geometry> geometry(ids.size());
    VisitFaceLoops(meshes, pipelines, ids, [&](uint32_t i, FaceLoops &&loops) { static_cast<FaceLoops &>(geometry[i]) = std::move(loops); });
    std::vector<RecordElements> faces;
    for (uint32_t i = 0u; i < ids.size(); ++i) faces.push_back({ids[i], geometry[i].Faces});
    for (uint32_t i = 0u; auto &normals : FaceNormals(meshes, pipelines, faces)) geometry[i++].Normals = std::move(normals);
    return geometry;
}
// The sums `a + Sign * b + Offset` of one task's point sets, set s holding points first_point[s] to first_point[s + 1].
// A negative sign reflects the second mesh, which is the brush an erosion subtracts.
struct HullSums {
    uint32_t First, Second;
    std::vector<uint32_t> FirstHandles, SecondHandles;
    std::vector<HullSumSet> Sets;
    std::vector<uint32_t> FirstPoint;
    float Sign;
    vec3 Offset;
    vec3 Facing;
};
// Each task's records of the convex hulls of its point sets, with every task's hulls in one quickhull.
std::vector<std::vector<uint32_t>> HullRecords(state::Registry &r, std::span<const HullSums> tasks) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    mtl::ComputeChain chain{meshes.BufferContext()};
    std::vector<uint32_t> first_point{0u};
    for (const auto &task : tasks) {
        const auto base = first_point.back();
        for (size_t s = 1u; s < task.FirstPoint.size(); ++s) first_point.push_back(base + task.FirstPoint[s]);
    }
    const auto points = first_point.back(), sets = uint32_t(first_point.size() - 1u);
    const auto positions = chain.Scratch.Allocate(3u * (points + sets));
    std::vector<HullPointsJob> jobs;
    for (uint32_t base = 0u; const auto &task : tasks) {
        jobs.push_back({
            .Positions = {chain.Scratch.Buffer.Slot, positions.Offset + 3u * base},
            .Count = task.FirstPoint.back(),
            .VertexSlot = meshes.Arenas().Vertices.Buffer.Slot,
            .First = meshes.GetConnectivityRef(task.First),
            .Second = meshes.GetConnectivityRef(task.Second),
            .FirstHandles = chain.Upload(std::as_bytes(std::span{task.FirstHandles})),
            .SecondHandles = chain.Upload(std::as_bytes(std::span{task.SecondHandles})),
            .Sets = chain.Upload(std::as_bytes(std::span{task.Sets})),
            .SetCount = uint32_t(task.Sets.size()),
            .SecondSign = task.Sign,
            .SecondOffset = task.Offset,
        });
        base += task.FirstPoint.back();
    }
    mtl::GatherPass(chain, GetMeshPipelines(r)[MeshPass::HullSums], std::span{jobs}, &HullPointsJob::Count);
    const auto hulls = sets ? ConvexHulls(r, chain, positions.Offset, first_point) : std::vector<std::vector<std::vector<uint32_t>>>{};
    const auto sums = ReadScratch<vec3>(chain, positions.Offset, points);
    std::vector<MeshSource> sources;
    std::vector<uint32_t> owners;
    for (uint32_t t = 0u, s = 0u; t < tasks.size(); ++t) {
        for (size_t local = 0u; local + 1u < tasks[t].FirstPoint.size(); ++local, ++s) {
            if (hulls[s].empty()) continue;
            MeshData data;
            std::vector<uint32_t> index(first_point[s + 1u] - first_point[s], InvalidOffset);
            std::vector<uint32_t> loop;
            for (const auto &face : hulls[s]) {
                loop.clear();
                for (const auto v : face) {
                    if (index[v] == InvalidOffset) {
                        index[v] = uint32_t(data.Positions.size());
                        data.Positions.push_back(sums[first_point[s] + v]);
                    }
                    loop.push_back(index[v]);
                }
                if (hulls[s].size() == 1u && Dot(tasks[t].Facing, tasks[t].Facing) > 0.f) {
                    vec3 normal{};
                    for (size_t i = 1; i + 1 < loop.size(); ++i) normal += Cross(data.Positions[loop[i]] - data.Positions[loop[0]], data.Positions[loop[i + 1]] - data.Positions[loop[0]]);
                    if (Dot(normal, tasks[t].Facing) < 0.f) std::ranges::reverse(loop);
                }
                data.AddFace(loop);
            }
            sources.push_back({.Data = std::move(data)});
            owners.push_back(t);
        }
    }
    std::vector<std::vector<uint32_t>> records(tasks.size());
    if (sources.empty()) return records;
    for (uint32_t i = 0u; const auto &created : CreateMeshes(r, sources)) records[owners[i++]].push_back(created.StoreId);
    ShadeNewMeshes(r);
    return records;
}
// Two records to combine, the second moved by Offset, as solids or as regions in a plane.
struct Combination {
    const std::optional<RegionPlane> &Plane;
    uint32_t First, Second;
    BooleanOperation Operation;
    vec3 Offset{};
};
// Each combination's new record, with every solid combination in one batch.
std::vector<std::optional<uint32_t>> Combine(state::Registry &r, std::span<const Combination> combinations) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    std::vector<std::optional<uint32_t>> combined(combinations.size());
    std::vector<MeshTopologyTask> tasks;
    std::vector<RegionTask> regions;
    std::vector<uint32_t> tasked, regioned;
    for (uint32_t i = 0u; i < combinations.size(); ++i) {
        const auto &[plane, first, second, operation, offset] = combinations[i];
        if (!plane) {
            if (auto task = BooleanTask(Mesh{meshes, first}, Mesh{meshes, second}, operation, mat3{1.f}, offset)) {
                tasks.push_back(std::move(*task));
                tasked.push_back(i);
            }
            continue;
        }
        const auto fill = operation == BooleanOperation::Union ? RegionFill::Union : operation == BooleanOperation::Difference ? RegionFill::Difference : RegionFill::Intersection;
        regions.push_back({*plane, {RegionOperand{.Record = first}, RegionOperand{.Record = second, .Translation = offset}}, fill});
        regioned.push_back(i);
    }
    // The region combinations fill together and their records create in one batch.
    std::vector<MeshSource> sources;
    std::vector<uint32_t> filled;
    for (uint32_t k = 0u; const auto &loops : RegionLoops(r, regions)) {
        if (loops && !loops->empty()) {
            sources.push_back({.Data = RegionData(regions[k].Plane, *loops)});
            filled.push_back(regioned[k]);
        }
        ++k;
    }
    if (!sources.empty()) {
        for (uint32_t k = 0u; const auto &created : CreateMeshes(r, sources)) combined[filled[k++]] = created.StoreId;
        ShadeNewMeshes(r);
    }
    if (tasks.empty()) return combined;
    for (uint32_t k = 0u; const auto record : ExecuteGeometryBooleans(r, tasks)) combined[tasked[k++]] = record;
    return combined;
}
// A record moved by Offset that combines with a fold's result.
struct Term {
    uint32_t Record;
    BooleanOperation Operation;
    vec3 Offset;
};
// A task's hulls, united in order and then united with or subtracted from its first record.
// The result then combines with each term in order and moves by Move.
struct Fold {
    uint32_t Task, First;
    std::vector<uint32_t> Hulls;
    bool Erosion;
    vec3 Move;
    std::optional<RegionPlane> Plane;
    std::vector<Term> Terms{};
    std::optional<uint32_t> Swept{};
};
// Folds every task's hulls, the k-th hull of every task in one batch, and returns each task's result.
std::vector<std::optional<uint32_t>> FoldHulls(state::Registry &r, std::vector<Fold> &folds) {
    auto &meshes = r.Resources.get<MeshStore>();
    for (size_t k = 0u; std::ranges::any_of(folds, [k](const Fold &fold) { return k < fold.Hulls.size(); }); ++k) {
        std::vector<Combination> unions;
        std::vector<Fold *> united;
        for (auto &fold : folds) {
            if (k >= fold.Hulls.size()) continue;
            if (!fold.Swept) {
                fold.Swept = fold.Hulls[k];
                continue;
            }
            unions.push_back({fold.Plane, *fold.Swept, fold.Hulls[k], BooleanOperation::Union});
            united.push_back(&fold);
        }
        for (uint32_t i = 0u; const auto joined : Combine(r, unions)) {
            auto &fold = *united[i];
            meshes.Release(fold.Hulls[k]);
            meshes.Release(*fold.Swept);
            fold.Swept = joined;
            // A union of non-empty operands is never empty, so a union without a result ends its fold without one.
            if (!joined) {
                for (size_t rest = k + 1u; rest < fold.Hulls.size(); ++rest) meshes.Release(fold.Hulls[rest]);
                fold.Hulls.resize(k + 1u);
            }
            ++i;
        }
    }
    std::vector<Combination> finals;
    std::vector<uint32_t> finished;
    for (uint32_t f = 0u; f < folds.size(); ++f) {
        if (!folds[f].Swept) continue;
        finals.push_back({folds[f].Plane, folds[f].First, *folds[f].Swept, folds[f].Erosion ? BooleanOperation::Difference : BooleanOperation::Union});
        finished.push_back(f);
    }
    std::vector<std::optional<uint32_t>> results(folds.size());
    for (uint32_t i = 0u; const auto result : Combine(r, finals)) {
        meshes.Release(*folds[finished[i]].Swept);
        results[finished[i++]] = result;
    }
    // The k-th term of every fold combines in one batch.
    for (size_t k = 0u;; ++k) {
        std::vector<Combination> combinations;
        std::vector<uint32_t> combined;
        for (uint32_t f = 0u; f < folds.size(); ++f) {
            if (!results[f] || k >= folds[f].Terms.size()) continue;
            const auto &[record, operation, offset] = folds[f].Terms[k];
            combinations.push_back({folds[f].Plane, *results[f], record, operation, offset});
            combined.push_back(f);
        }
        if (combinations.empty()) break;
        for (uint32_t i = 0u; const auto result : Combine(r, combinations)) {
            meshes.Release(*results[combined[i]]);
            results[combined[i++]] = result;
        }
    }
    std::vector<std::pair<uint32_t, vec3>> moved;
    for (uint32_t f = 0u; f < folds.size(); ++f) {
        if (results[f] && folds[f].Move != vec3{}) moved.emplace_back(*results[f], folds[f].Move);
    }
    if (moved.empty()) return results;
    mtl::ComputeChain chain{meshes.BufferContext()};
    for (const auto &[id, offset] : moved) {
        const PositionOperationTarget target{.StoreId = id, .Every = true};
        (void)EncodePositionOperations(meshes, GetMeshPipelines(r), chain, target, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = offset}}});
    }
    EncodeRecordRefresh(r, chain, moved | std::views::keys | std::ranges::to<std::vector>());
    chain.Submit();
    return results;
}
} // namespace

std::vector<std::optional<uint32_t>> MinkowskiRecords(state::Registry &r, std::span<const MinkowskiTask> tasks) {
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &pipelines = GetMeshPipelines(r);
    std::vector<std::optional<uint32_t>> results(tasks.size());
    std::vector<uint32_t> active, ids;
    for (uint32_t t = 0u; t < tasks.size(); ++t) {
        if (!Mesh{meshes, tasks[t].First}.FaceCount() || !Mesh{meshes, tasks[t].Second}.FaceCount()) continue;
        active.push_back(t);
        ids.insert(ids.end(), {tasks[t].First, tasks[t].Second});
    }
    const auto geometry = ReadGeometry(r, ids);
    // Each active task's operands after the swap, its plane, whether each is convex, and whether it is a sum of convex operands.
    struct Operands {
        uint32_t Task, A, B;
        const Geometry *AGeometry, *BGeometry;
        std::optional<RegionPlane> Plane;
        bool BConvex, ConvexSum;
    };
    std::vector<Operands> operands;
    for (uint32_t i = 0u; i < active.size(); ++i) {
        const auto [first, second, operation] = tasks[active[i]];
        const Mesh first_mesh{meshes, first}, second_mesh{meshes, second};
        const auto plane = first_mesh.HasClosedSurface() && second_mesh.HasClosedSurface() ? std::nullopt : FacePlane(r, first_mesh);
        Operands o{active[i], first, second, &geometry[2u * i], &geometry[2u * i + 1u], plane};
        bool a_convex = plane ? o.AGeometry->ConvexRegion(*plane) : o.AGeometry->Convex(first_mesh);
        o.BConvex = plane ? o.BGeometry->ConvexRegion(*plane) : o.BGeometry->Convex(second_mesh);
        // A sum commutes, so the convex operand is the one swept along the other's faces.
        if (operation == MinkowskiOperation::Sum && a_convex && !o.BConvex) {
            std::swap(o.A, o.B);
            std::swap(o.AGeometry, o.BGeometry);
            std::swap(a_convex, o.BConvex);
        }
        o.ConvexSum = operation == MinkowskiOperation::Sum && a_convex && o.BConvex;
        operands.push_back(std::move(o));
    }
    // The triangles of the solid operands that sweep by their triangles.
    std::vector<uint32_t> triangulated;
    for (const auto &o : operands) {
        if (o.ConvexSum || o.Plane) continue;
        triangulated.push_back(o.A);
        if (!o.BConvex) triangulated.push_back(o.B);
    }
    const auto triangles = TriangleVertices(meshes, pipelines, triangulated);
    std::vector<HullSums> sums;
    std::vector<Fold> folds;
    std::vector<MeshSource> reflections;
    std::vector<uint32_t> reflected;
    for (uint32_t next = 0u; const auto &o : operands) {
        const Mesh a{meshes, o.A}, b{meshes, o.B};
        const auto facing = o.Plane ? o.Plane->Normal : vec3{};
        if (o.ConvexSum) {
            auto a_handles = HandleList(a.vertices()), b_handles = HandleList(b.vertices());
            const auto count = uint32_t(a_handles.size()), b_count = uint32_t(b_handles.size());
            sums.push_back({o.A, o.B, std::move(a_handles), std::move(b_handles), {HullSumSet{.CountB = b_count}}, {0u, count * b_count}, 1.f, vec3{}, facing});
            continue;
        }
        const uint32_t k = o.Plane ? 2u : 3u;
        auto a_simplices = o.Plane ? o.AGeometry->EdgeHandles() : triangles[next++];
        std::vector<HullSumSet> sets;
        std::vector<uint32_t> first_point{0u};
        std::vector<uint32_t> b_handles;
        if (o.BConvex) {
            b_handles = HandleList(b.vertices());
            const auto b_count = uint32_t(b_handles.size());
            for (uint32_t sa = 0u; sa < a_simplices.size(); sa += k) {
                sets.push_back({.First = first_point.back(), .FirstA = sa, .CountB = b_count});
                first_point.push_back(first_point.back() + k * b_count);
            }
        } else {
            b_handles = o.Plane ? o.BGeometry->EdgeHandles() : triangles[next++];
            for (uint32_t sa = 0u; sa < a_simplices.size(); sa += k) {
                const auto na = o.Plane ? vec3{} : o.AGeometry->TriangleNormal(std::span{a_simplices}.subspan(sa, 3u));
                for (uint32_t sb = 0u; sb < b_handles.size(); sb += k) {
                    // Parallel faces sweep a flat hull, which adds nothing.
                    if (!o.Plane && std::abs(std::abs(Dot(na, o.BGeometry->TriangleNormal(std::span{b_handles}.subspan(sb, 3u)))) - 1.f) < 1e-6f) continue;
                    sets.push_back({.First = first_point.back(), .FirstA = sa, .FirstB = sb, .CountB = k});
                    first_point.push_back(first_point.back() + k * k);
                }
            }
        }
        // For closed A and B, the sum A + B is the sweep of A's surface by B's surface, united with A moved by one point of each component of B's surface and with B moved by one point of each component of A's surface.
        // The erosion of A by B is the intersection of A moved by minus one point of each component of B's surface, less the sweep of A's surface by the reflected surface of B, less the reflected B moved to one point of each component of A's surface.
        // Any point of B's solid joined to a component of B's surface serves as that component's point.
        // The hulls sweep relative to a point p of B, so the first record in place stands for the term of p's component, and the result moves by p, negated for an erosion.
        // p is the point the construction holds already, the origin for a sum and the vertex average for an erosion, when exactly one component winds around it, and otherwise the vertex of B's first component.
        // Each other component of B's surface adds its translate of A.
        // A convex B sweeps as a solid, which holds B moved to every point of A's surface, so it adds no translate of B.
        // A component of A's surface adds its translate of B only when B's bounds cover the component's extent, which every translate of B holding the component requires.
        const auto erosion = tasks[o.Task].Operation == MinkowskiOperation::Difference;
        const float sign = erosion ? -1.f : 1.f;
        // A convex B holds its vertex average, so an erosion by it reads no components.
        const auto brush = erosion && o.BConvex ? Components{} : ConnectedComponents(*o.BGeometry);
        const auto held = erosion ? o.BGeometry->Centroid(b) : vec3{};
        const auto winds = Winds(*o.BGeometry, brush, held, o.Plane);
        const bool in_place = brush.Of.empty() || std::ranges::count(winds, true) == 1;
        const auto point = in_place ? held : o.BGeometry->Positions[brush.Of.front().Vertex];
        sums.push_back({o.A, o.B, std::move(a_simplices), std::move(b_handles), std::move(sets), std::move(first_point), sign, erosion ? point : vec3{} - point, facing});
        auto &fold = folds.emplace_back(Fold{.Task = o.Task, .First = o.A, .Erosion = erosion, .Move = point * sign, .Plane = o.Plane});
        for (uint32_t k = 0u; k < brush.Of.size(); ++k) {
            if (in_place ? winds[k] : k == 0u) continue;
            fold.Terms.push_back({o.A, erosion ? BooleanOperation::Intersection : BooleanOperation::Union, (o.BGeometry->Positions[brush.Of[k].Vertex] - point) * sign});
        }
        if (o.BConvex) continue;
        auto lo = brush.Of.front().Min, hi = brush.Of.front().Max;
        for (const auto &component : brush.Of) {
            lo = Min(lo, component.Min);
            hi = Max(hi, component.Max);
        }
        const auto first_term = fold.Terms.size();
        const auto cover = hi - lo;
        for (const auto &component : ConnectedComponents(*o.AGeometry).Of) {
            const auto extent = component.Max - component.Min;
            if (extent.x > cover.x || extent.y > cover.y || extent.z > cover.z) continue;
            fold.Terms.push_back({o.B, erosion ? BooleanOperation::Difference : BooleanOperation::Union, o.AGeometry->Positions[component.Vertex] - point * sign});
        }
        if (erosion && fold.Terms.size() > first_term) {
            reflections.push_back(Reflection(*o.BGeometry, !o.Plane));
            reflected.push_back(uint32_t(folds.size() - 1u));
        }
    }
    // An erosion's terms of B take the reflected brush.
    std::vector<uint32_t> brushes;
    if (!reflections.empty()) {
        for (uint32_t i = 0u; const auto &created : CreateMeshes(r, reflections)) {
            for (auto &term : folds[reflected[i++]].Terms) {
                if (term.Operation == BooleanOperation::Difference) term.Record = created.StoreId;
            }
            brushes.push_back(created.StoreId);
        }
        ShadeNewMeshes(r);
    }
    // An operand's words are those of its hull records, each at most a record of its set's points triangulated.
    const auto words = sums | std::views::transform([&](const HullSums &sum) {
        uint64_t total = 0u;
        for (size_t s = 1u; s < sum.FirstPoint.size(); ++s) {
            const auto points = sum.FirstPoint[s] - sum.FirstPoint[s - 1u];
            total += meshes.RecordWords({points, 6u * points, 2u * points});
        }
        return uint32_t(std::min<uint64_t>(total, UINT32_MAX));
    }) | std::ranges::to<std::vector>();
    // The operands sweep and fold in chunks whose hull records stay within the budget, in order.
    for (uint32_t f = 0u; const auto chunk : ChunkByScratch(uint32_t(sums.size()), ScratchWordBudget, [&](uint32_t i) { return words[i]; }).Chunks) {
        const auto hulls = HullRecords(r, std::span{sums}.subspan(chunk.Offset, chunk.Count));
        std::vector<Fold> folding;
        for (uint32_t i = 0u; i < chunk.Count; ++i) {
            const auto &o = operands[chunk.Offset + i];
            if (o.ConvexSum) {
                results[o.Task] = hulls[i].empty() ? std::nullopt : std::optional{hulls[i].front()};
                continue;
            }
            auto &fold = folding.emplace_back(std::move(folds[f++]));
            fold.Hulls = hulls[i];
        }
        for (uint32_t k = 0u; const auto result : FoldHulls(r, folding)) results[folding[k++].Task] = result;
    }
    for (const auto id : brushes) r.Resources.get<MeshStore>().Release(id);
    return results;
}
