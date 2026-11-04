#pragma once

#include "MeshProperties.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/TopologyOperations.h"
#include "procedure/Procedure.h"
#include "state/Registry.h"

// The topology operator table over a patchwork region.
namespace test {
// A 4x4 grid with a nonplanar cell, a concave cell, a triangulated cell and a short edge, then an isolated vertex and a loose edge.
inline MeshData Patchwork() {
    MeshData data;
    for (uint32_t y = 0u; y < 5u; ++y)
        for (uint32_t x = 0u; x < 5u; ++x) data.Positions.push_back({float(x), float(y), 0.f});
    data.Positions[6].z = .5f;
    data.Positions[12] = {2.9f, 2.9f, 0.f};
    data.Positions[13] = {3.f, 1.05f, 0.f};
    for (uint32_t y = 0u; y < 4u; ++y)
        for (uint32_t x = 0u; x < 4u; ++x) {
            const auto v = y * 5u + x;
            if (x == 1u && y == 2u) {
                data.AddFace(std::array{v, v + 1u, v + 5u});
                data.AddFace(std::array{v + 1u, v + 6u, v + 5u});
            } else data.AddFace(std::array{v, v + 1u, v + 6u, v + 5u});
        }
    data.Positions.insert(data.Positions.end(), {{6.f, 6.f, 0.f}, {6.f, 0.f, 0.f}, {7.f, 0.f, 0.f}});
    data.Edges.push_back({26u, 27u});
    return data;
}
// The vertices `inside` accepts, and the edges and faces whose vertices it all accepts.
inline GeometrySelection SelectionWhere(const Mesh &mesh, auto &&inside) {
    GeometrySelection selection;
    for (const auto v : mesh.vertices())
        if (inside(v)) selection.Vertices.push_back(*v);
    for (const auto e : mesh.edges())
        if (std::ranges::all_of(EdgeVertices(mesh, e), inside)) selection.Edges.push_back(*e);
    for (const auto face : mesh.faces())
        if (std::ranges::all_of(FaceVertices(mesh, face), inside)) selection.Faces.push_back(*face);
    return selection;
}
// The inner 3x3 grid vertices, the isolated vertex and the loose edge, with every edge and face among them.
inline GeometrySelection Region(const Mesh &mesh) {
    const auto inside = [&](he::VH v) {
        const auto i = mesh.VertexOrdinal(v);
        return i >= 25u || (i % 5u >= 1u && i % 5u <= 3u && i / 5u >= 1u && i / 5u <= 3u);
    };
    auto result = SelectionWhere(mesh, inside);
    for (auto *handles : {&result.Vertices, &result.Edges, &result.Faces}) std::ranges::sort(*handles);
    return result;
}

// The faces whose vertices all lie between y_min and y_max each have no vertex on both sides of the plane at x.
inline bool Uncrossed(const Mesh &mesh, float x, float y_min, float y_max) {
    for (const auto face : mesh.faces()) {
        float lo = 1e30f, hi = -1e30f, y_lo = 1e30f, y_hi = -1e30f;
        for (const auto v : FaceVertices(mesh, face)) {
            const auto p = Position(mesh, v);
            lo = std::min(lo, p.x);
            hi = std::max(hi, p.x);
            y_lo = std::min(y_lo, p.y);
            y_hi = std::max(y_hi, p.y);
        }
        if (y_lo >= y_min && y_hi <= y_max && lo < x - 1e-5f && hi > x + 1e-5f) return false;
    }
    return true;
}

// A topology operator over the region: a task taking the region selection, or the planner that makes it from the region.
// Kind is the node kind whose cook plans the task through PlanTopology, or none for an operator without such a node.
struct TopologyCase {
    std::string_view Name;
    std::optional<uint16_t> Kind;
    MeshTopologyTask Task;
    std::optional<MeshTopologyTask> (*Plan)(state::Registry &, const Mesh &, const GeometrySelection &){};
    bool (*Holds)(const Mesh &){};
};
inline std::optional<MeshTopologyTask> PlanCase(const TopologyCase &c, state::Registry &r, const Mesh &mesh) {
    const auto selection = Region(mesh);
    auto task = c.Plan ? c.Plan(r, mesh, selection) : std::optional{c.Task};
    if (!task) return task;
    task->SourceId = mesh.GetStoreId();
    if (!c.Plan) task->Selection = selection;
    return task;
}
inline MeshTopologyTask Op(MeshTopologyOp op, float param0 = 0.f, uint32_t flags = 0u, uint32_t steps = 1u) {
    return {.SourceId = InvalidOffset, .Op = op, .Param0 = param0, .Flags = flags, .Steps = steps};
}

inline const std::array<TopologyCase, 42> TopologyCases = [] {
    using enum MeshTopologyOp;
    using procedure::KindOf;
    namespace m = action::mesh;
    return std::array<TopologyCase, 42>{{
        {"DeleteVertices", KindOf<m::Delete>, Op(DeleteVertices)},
        {"DeleteEdges", KindOf<m::Delete>, Op(DeleteEdges)},
        {"DeleteFaces", KindOf<m::Delete>, Op(DeleteFaces)},
        {"DeleteOnlyEdgesFaces", KindOf<m::Delete>, Op(DeleteOnlyEdgesFaces)},
        {"DeleteOnlyFaces", KindOf<m::Delete>, Op(DeleteOnlyFaces)},
        // Hidden vertices stay operands of a headless edit, so every loose element goes.
        {"DeleteLoose", KindOf<m::Delete>, {.Op = DeleteLoose}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
             auto &meshes = r.Resources.get<MeshStore>();
             meshes.EditHiddenBlocks(Element::Vertex, meshes.Arenas().Vertices.Entries(meshes.Get(mesh.GetStoreId()).Vertices), [](uint32_t, auto &words) { std::ranges::fill(words, ~0u); });
             return std::optional{MeshTopologyTask{.Op = DeleteLoose, .Selection = selection}}; }, [](const Mesh &mesh) {
             const auto &c = mesh.GetConnectivity();
             for (const auto e : mesh.edges())
                 if (!c.FaceOf(Halfedge(mesh, e))) return false;
             for (const auto v : mesh.vertices())
                 if (c.Incoming(v).y == 0u) return false;
             return true; }},
        {"DissolveVertices", KindOf<m::Dissolve>, Op(DissolveVertices)},
        {"DissolveEdges", KindOf<m::Dissolve>, Op(DissolveEdges)},
        {"DissolveFaces", KindOf<m::Dissolve>, Op(DissolveFaces)},
        {"DissolveLimited", KindOf<m::Dissolve>, Op(DissolveLimited, .3f)},
        // Moving vertex 7 onto vertex 8 makes a run of short edges from 7 through 8 to 13, which one pass merges into one vertex.
        {"DissolveDegenerate", KindOf<m::Dissolve>, {.Op = DissolveDegenerate}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
             const PositionOperationTarget moving{.StoreId = mesh.GetStoreId(), .Selection = GeometrySelection{.Vertices = {*mesh.VertexAt(7u)}}};
             ExecuteGeometryPositions(r, moving, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {1.f, 0.f, 0.f}}}});
             return std::optional{MeshTopologyTask{.Op = DissolveDegenerate, .Param0 = .1f, .Selection = selection}}; }, [](const Mesh &mesh) {
             for (const auto e : mesh.edges()) {
                 const auto [from, to] = EdgeVertices(mesh, e);
                 if (Length(Position(mesh, from) - Position(mesh, to)) < .1f) return false;
             }
             return true; }},
        // A merge at a target merges onto the selection's first vertex at its center, as the Center mode does.
        {"MergeAtTarget", KindOf<m::Merge>, {.Op = MergeAtTarget}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(r, mesh, selection, GeometryMergeMode::Center, 0.f, selection.Vertices.front()); }},
        {"MergeByDistance", KindOf<m::Merge>, {.Op = MergeByDistance}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(r, mesh, selection, GeometryMergeMode::ByDistance, .1f, InvalidOffset); }},
        {"MergeCollapse", KindOf<m::Merge>, {.Op = MergeCollapse}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(r, mesh, selection, GeometryMergeMode::Collapse, 0.f, InvalidOffset); }},
        {"RotateEdges", {}, {.Op = RotateEdges}, [](state::Registry &, const Mesh &mesh, const GeometrySelection &selection) {
             GeometrySelection interior;
             for (const auto edge : selection.Edges) {
                 const auto h = Halfedge(mesh, he::EH{edge}), opposite = Opposite(mesh, h);
                 if (opposite && std::ranges::binary_search(selection.Faces, *Face(mesh, h)) && std::ranges::binary_search(selection.Faces, *Face(mesh, opposite))) interior.Edges.push_back(edge);
             }
             if (interior.Edges.empty()) return std::optional<MeshTopologyTask>{};
             return std::optional{MeshTopologyTask{.Op = MeshTopologyOp::RotateEdges, .SelectionElement = Element::Edge, .Selection = std::move(interior)}};
         }},
        // A face replacement is the area beautify that plans it.
        {"ReplaceFaces", KindOf<m::BeautifyFaces>, {.Op = ReplaceFaces}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) { return BeautifyFaceTask(r, mesh, selection); }, [](const Mesh &mesh) { return Delaunay(mesh, AllHandles(mesh).Faces); }},
        {"AddPrimitives", KindOf<m::ConvexHull>, {.Op = AddPrimitives}, [](state::Registry &r, const Mesh &mesh, const GeometrySelection &selection) {
             auto tasks = ConvexHullTasks(r, mesh, selection);
             const auto faces = std::ranges::find(tasks, MeshTopologyOp::AddPrimitives, &MeshTopologyTask::Op);
             if (faces == tasks.end()) return std::optional<MeshTopologyTask>{};
             faces->Selection = selection;
             return std::optional{std::move(*faces)};
         }},
        {"Wireframe", KindOf<m::Wireframe>, Op(Wireframe, .05f)},
        {"BevelVertices", KindOf<m::Bevel>, Op(BevelVertices, .1f)},
        {"BevelEdges", KindOf<m::Bevel>, Op(BevelEdges, .1f)},
        {"TrisToQuads", KindOf<m::TrisToQuads>, Op(TrisToQuads)},
        {"ConnectVertices", KindOf<m::ConnectVertices>, Op(ConnectVertices)},
        {"SplitGeometry", KindOf<m::Split>, Op(SplitGeometry)},
        {"DuplicateGeometry", KindOf<m::Duplicate>, Op(DuplicateGeometry)},
        {"ExtrudeVertices", KindOf<procedure::Extrude>, Op(ExtrudeVertices)},
        {"ExtrudeEdges", KindOf<procedure::Extrude>, Op(ExtrudeEdges)},
        {"ExtrudeRegion", KindOf<procedure::Extrude>, Op(ExtrudeRegion)},
        {"ExtrudeFacesIndividual", KindOf<procedure::Extrude>, Op(ExtrudeFacesIndividual)},
        {"Subdivide", KindOf<m::Subdivide>, Op(Subdivide, 1.f)},
        // Clip space spans the 4 by 4 grid, so the screen line 37.5 pixels into a 100 pixel extent cuts its faces at x = 1.5.
        {"Knife", {}, {.Op = Subdivide}, [](state::Registry &, const Mesh &mesh, const GeometrySelection &) {
             const mat4 grid_to_clip{vec4{.5f, 0.f, 0.f, 0.f}, vec4{0.f, .5f, 0.f, 0.f}, vec4{0.f, 0.f, 1.f, 0.f}, vec4{-1.f, -1.f, 0.f, 1.f}};
             return std::optional{KnifeTask(mesh, grid_to_clip, {100.f, 100.f}, {37.5f, 0.f}, {37.5f, 100.f})}; }, [](const Mesh &mesh) { return Uncrossed(mesh, 1.5f, 1.f, 3.f); }},
        // The plane through the grid's x = 2 column crosses faces at their vertices and through their interiors.
        {"PlaneCut", {}, {.Op = Subdivide}, [](state::Registry &, const Mesh &mesh, const GeometrySelection &) { return std::optional{PlaneCutTask(mesh, {2.f, 0.f, 0.f}, {1.f, 0.f, 0.f})}; }, [](const Mesh &mesh) { return Uncrossed(mesh, 2.f, -1e30f, 1e30f); }},
        {"EdgeSplit", KindOf<m::EdgeSplit>, Op(EdgeSplit, 0.f, TopologyFlagSelectCopies)},
        {"Triangulate", KindOf<m::SplitFaces>, Op(Triangulate)},
        {"SplitNonplanarFaces", KindOf<m::SplitFaces>, Op(SplitNonplanarFaces, .01f)},
        {"SplitConcaveFaces", KindOf<m::SplitFaces>, Op(SplitConcaveFaces)},
        {"Poke", KindOf<m::Poke>, Op(Poke)},
        {"FlipNormals", KindOf<m::FlipNormals>, Op(FlipNormals)},
        {"InsetRegion", KindOf<m::Inset>, Op(InsetRegion, .1f)},
        {"InsetIndividual", KindOf<m::Inset>, Op(InsetIndividual, .1f)},
        {"Solidify", KindOf<m::Solidify>, Op(Solidify, .1f)},
        {"KeepSelectedFaces", {}, {.Op = KeepSelectedFaces, .SelectionElement = Element::Face}},
        {"ExtrudeSteps", KindOf<procedure::Extrude>, {.Op = ExtrudeRegion}, [](state::Registry &, const Mesh &mesh, const GeometrySelection &selection) { return std::optional{ExtrudeStepsTask(mesh, selection, Element::None, 2u, mat3{1.f}, {0.f, 0.f, 1.f}, {})}; }},
    }};
}();
} // namespace test
