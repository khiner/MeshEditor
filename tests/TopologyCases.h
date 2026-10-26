#pragma once

#include "mesh/BeautifyFaces.h"
#include "mesh/Decimate.h"
#include "mesh/Mesh.h"
#include "mesh/MeshData.h"
#include "mesh/TopologyOperations.h"

// The topology operator table over a patchwork region, which the headless kernel tests and the procedure node tests share.
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
// The inner 3x3 grid vertices, the isolated vertex and the loose edge, with every edge and face among them.
inline GeometrySelection Region(const Mesh &mesh) {
    const auto inside = [&](he::VH v) {
        const auto i = mesh.VertexOrdinal(v);
        return i >= 25u || (i % 5u >= 1u && i % 5u <= 3u && i / 5u >= 1u && i / 5u <= 3u);
    };
    GeometrySelection result;
    for (const auto v : mesh.vertices())
        if (inside(v)) result.Vertices.push_back(*v);
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        if (inside(mesh.GetFromVertex(h)) && inside(mesh.GetToVertex(h))) result.Edges.push_back(*e);
    }
    for (const auto f : mesh.faces())
        if (std::ranges::all_of(mesh.fv_range(f), inside)) result.Faces.push_back(*f);
    for (auto *handles : {&result.Vertices, &result.Edges, &result.Faces}) std::ranges::sort(*handles);
    return result;
}

// A task taking the region selection, or the planner that makes it from the region.
struct TopologyCase {
    std::string_view Name;
    MeshTopologyTask Task;
    std::optional<MeshTopologyTask> (*Plan)(const MeshStore &, const Mesh &, const GeometrySelection &){};
};
inline std::optional<MeshTopologyTask> PlanCase(const TopologyCase &c, const MeshStore &meshes, const Mesh &mesh) {
    const auto selection = Region(mesh);
    auto task = c.Plan ? c.Plan(meshes, mesh, selection) : std::optional{c.Task};
    if (!task) return task;
    task->SourceId = mesh.GetStoreId();
    if (!c.Plan || task->Op == MeshTopologyOp::AddPrimitives) task->Selection = selection;
    return task;
}
inline MeshTopologyTask Op(MeshTopologyOp op, float param0 = 0.f, uint32_t flags = 0u, uint32_t steps = 1u, Element element = Element::None) {
    return {.SourceId = InvalidOffset, .Op = op, .Param0 = param0, .Flags = flags, .Steps = steps, .SelectionElement = element};
}

// Every topology operator, each changing the patchwork region.
inline const std::array<TopologyCase, 41> TopologyCases = [] {
    using enum MeshTopologyOp;
    return std::array<TopologyCase, 41>{{
        {"DeleteVertices", Op(DeleteVertices)},
        {"DeleteEdges", Op(DeleteEdges)},
        {"DeleteFaces", Op(DeleteFaces)},
        {"DeleteOnlyEdgesFaces", Op(DeleteOnlyEdgesFaces)},
        {"DeleteOnlyFaces", Op(DeleteOnlyFaces)},
        {"DeleteLoose", Op(DeleteLoose)},
        {"DissolveVertices", Op(DissolveVertices)},
        {"DissolveEdges", Op(DissolveEdges)},
        {"DissolveFaces", Op(DissolveFaces)},
        {"DissolveLimited", Op(DissolveLimited, .3f)},
        {"DissolveDegenerate", Op(DissolveDegenerate, .1f)},
        {"MergeAtTarget", {.Op = MeshTopologyOp::MergeAtTarget}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(meshes, mesh, selection, GeometryMergeMode::Center, 0.f, selection.Vertices.front()); }},
        {"MergeByDistance", {.Op = MeshTopologyOp::MergeByDistance}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(meshes, mesh, selection, GeometryMergeMode::ByDistance, .1f, InvalidOffset); }},
        {"MergeCollapse", {.Op = MeshTopologyOp::MergeCollapse}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return MergeTask(meshes, mesh, selection, GeometryMergeMode::Collapse, 0.f, InvalidOffset); }},
        {"Decimate", {.Op = MeshTopologyOp::Decimate}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return DecimateTask(meshes, mesh, selection, .5f); }},
        {"RotateEdges", {.Op = MeshTopologyOp::RotateEdges}, [](const MeshStore &, const Mesh &mesh, const GeometrySelection &selection) {
             // Edges between two selected faces rotate.
             GeometrySelection interior;
             for (const auto edge : selection.Edges) {
                 const auto h = mesh.GetHalfedge(he::EH{edge}, 0u), opposite = mesh.GetOppositeHalfedge(h);
                 if (opposite && std::ranges::binary_search(selection.Faces, *mesh.GetFace(h)) && std::ranges::binary_search(selection.Faces, *mesh.GetFace(opposite))) interior.Edges.push_back(edge);
             }
             return RotateEdgesTask(mesh, interior);
         }},
        {"ReplaceFaces", {.Op = MeshTopologyOp::ReplaceFaces}, [](const MeshStore &, const Mesh &mesh, const GeometrySelection &selection) { return BeautifyFaceTask(mesh, selection, false); }},
        {"AddPrimitives", {.Op = MeshTopologyOp::AddPrimitives}, [](const MeshStore &meshes, const Mesh &mesh, const GeometrySelection &selection) { return ConvexHullTask(meshes, mesh, selection); }},
        {"Wireframe", Op(Wireframe, .05f)},
        {"BevelVertices", Op(BevelVertices, .1f)},
        {"BevelEdges", Op(BevelEdges, .1f)},
        {"TrisToQuads", Op(TrisToQuads)},
        {"ConnectVertices", Op(ConnectVertices)},
        {"SplitGeometry", Op(SplitGeometry)},
        {"DuplicateGeometry", Op(DuplicateGeometry)},
        {"ExtrudeVertices", Op(ExtrudeVertices)},
        {"ExtrudeEdges", Op(ExtrudeEdges)},
        {"ExtrudeRegion", Op(ExtrudeRegion)},
        {"ExtrudeFacesIndividual", Op(ExtrudeFacesIndividual)},
        {"Subdivide", Op(Subdivide, 1.f)},
        {"EdgeSplit", Op(EdgeSplit, 0.f, TopologyFlagRipSelectCopies)},
        {"Triangulate", Op(Triangulate)},
        {"SplitNonplanarFaces", Op(SplitNonplanarFaces, .01f)},
        {"SplitConcaveFaces", Op(SplitConcaveFaces)},
        {"Poke", Op(Poke)},
        {"FlipNormals", Op(FlipNormals)},
        {"InsetRegion", Op(InsetRegion, .1f)},
        {"InsetIndividual", Op(InsetIndividual, .1f)},
        {"Solidify", Op(Solidify, .1f)},
        {"KeepSelectedFaces", Op(KeepSelectedFaces, 0.f, 0u, 1u, Element::Face)},
        {"ExtrudeSteps", {.Op = MeshTopologyOp::ExtrudeRegion, .Flags = TopologyFlagTransformCopies}, [](const MeshStore &, const Mesh &mesh, const GeometrySelection &selection) { return std::optional{ExtrudeStepsTask(mesh, selection, Element::None, 2u, mat3{1.f}, {0.f, 0.f, 1.f}, {})}; }},
    }};
}();
} // namespace test
