#include "Paths.h"
#include "RunSuites.h"
#include "TopologyCases.h"
#include "mesh/BeautifyFaces.h"
#include "mesh/Decimate.h"
#include "mesh/GeometryRefresh.h"
#include "mesh/Mesh.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshPipelines.h"
#include "mesh/MeshTopology.h"
#include "mesh/SharpnessOperations.h"
#include "mesh/SpatialFaceWork.h"
#include "mesh/TopologyOperations.h"
#include "mesh/Unsubdivide.h"
#include "metal/Dispatch.h"
#include "metal/MetalCpp.h"
#include "metal/Shader.h"
#include "state/Scene.h"
#include <cmath>
#include <stdexcept>

using namespace boost::ut;

namespace {
using ::test::Op, ::test::Patchwork, ::test::PlanCase, ::test::Region, ::test::TopologyCase, ::test::TopologyCases;

struct GeometryFixture {
    NS::SharedPtr<NS::AutoreleasePool> Pool{NS::TransferPtr(NS::AutoreleasePool::alloc()->init())};
    mtl::Context Context;
    mtl::BindlessSet Slots{Context};
    mtl::BufferContext Buffers{Context, Slots};
    state::Scene Scene;
    GeometryFixture() {
        Scene.Context.emplace<mtl::LibraryCache>(Context, Paths::Shaders());
        Scene.Context.emplace<MeshStore>(Buffers);
    }
    MeshStore &Store() { return Scene.Context.get<MeshStore>(); }
    Mesh Get(uint32_t id) { return Mesh{Store(), id}; }
    uint32_t Create(MeshData data) { return CreateMesh(Scene, {.Data = std::move(data)}).StoreId; }
    GeometryTopologyResult Run(MeshTopologyTask task) { return ExecuteGeometryTopology(Scene, std::span{&task, 1u}).front(); }
};

MeshData Quad() {
    return MeshData{{{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}}, {{0u, 1u, 2u, 3u}}};
}
GeometrySelection All(const Mesh &mesh) {
    GeometrySelection result;
    for (const auto v : mesh.vertices()) result.Vertices.push_back(*v);
    for (const auto e : mesh.edges()) result.Edges.push_back(*e);
    for (const auto f : mesh.faces()) result.Faces.push_back(*f);
    return result;
}

// Writes selections as mask blocks into an arena the test owns, one block run per domain with room for the blocks an edit inserts.
struct MaskWriter {
    BufferArena<SelectionBlock> Blocks;
    explicit MaskWriter(mtl::BufferContext &buffers) : Blocks{buffers, SlotType::Buffer} {}
    SelectionMasks Write(std::span<const uint32_t> handles, uint32_t capacity) {
        const auto range = Blocks.Allocate((capacity + MeshElementBlockSize - 1u) / MeshElementBlockSize + 64u);
        auto blocks = Blocks.GetMutable(range);
        std::ranges::fill(blocks, SelectionBlock{});
        for (const auto h : handles) blocks[h / MeshElementBlockSize][(h % MeshElementBlockSize) / 32u] |= 1u << (h % 32u);
        return {&Blocks, range.Offset};
    }
    GeometryMasks Write(const MeshStore &meshes, const GeometrySelection &selection) {
        const auto &a = meshes.Arenas();
        return {Write(selection.Vertices, a.Vertices.Capacity()), Write(selection.Edges, a.EdgeHalfedges.Capacity()), Write(selection.Faces, a.FaceTriangles.Capacity())};
    }
};

// Positions rounded so that results compare across meshes whose handles differ.
using Point = std::array<int64_t, 3>;
Point Key(vec3 p) { return {std::llround(p.x * 1e4f), std::llround(p.y * 1e4f), std::llround(p.z * 1e4f)}; }
std::vector<Point> Points(const Mesh &mesh) {
    std::vector<Point> points;
    for (const auto v : mesh.vertices()) points.push_back(Key(mesh.GetPosition(v)));
    return points;
}
// Each element's sorted corner points, sorted.
std::vector<std::vector<Point>> Shapes(const Mesh &mesh, Element element, std::span<const uint32_t> handles) {
    std::vector<std::vector<Point>> shapes;
    for (const auto handle : handles) {
        auto &shape = shapes.emplace_back();
        if (element == Element::Vertex) shape.push_back(Key(mesh.GetPosition(he::VH{handle})));
        else if (element == Element::Edge) {
            const auto h = mesh.GetHalfedge(he::EH{handle}, 0u);
            shape = {Key(mesh.GetPosition(mesh.GetFromVertex(h))), Key(mesh.GetPosition(mesh.GetToVertex(h)))};
        } else
            for (const auto v : mesh.fv_range(he::FH{handle})) shape.push_back(Key(mesh.GetPosition(v)));
        std::ranges::sort(shape);
    }
    std::ranges::sort(shapes);
    return shapes;
}
// The created, retained and output elements of each domain, as shapes.
struct Outcome {
    bool Changed, Copied;
    std::array<std::vector<std::vector<Point>>, 9> Elements;
    bool operator==(const Outcome &) const = default;
};
Outcome Describe(const MeshStore &meshes, uint32_t source, const GeometryTopologyResult &result) {
    const Mesh mesh{meshes, result.GeometryId};
    Outcome outcome{result.Changed, result.GeometryId != source};
    const auto all = All(mesh);
    for (uint32_t d = 0u; const auto element : {Element::Vertex, Element::Edge, Element::Face}) {
        outcome.Elements[d] = Shapes(mesh, element, result.Created.Get(element));
        outcome.Elements[3u + d] = Shapes(mesh, element, result.Retained.Get(element));
        outcome.Elements[6u + d++] = Shapes(mesh, element, all.Get(element));
    }
    return outcome;
}
} // namespace

void RegisterGeometryOperationsTests() {
    "headless topology uses explicit masks and returns created geometry"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selected = All(f.Get(id));
        expect(f.Store().Get(id).MeshletRoot == InvalidOffset);
        for (const auto op : {MeshTopologyOp::DuplicateGeometry, MeshTopologyOp::ExtrudeRegion, MeshTopologyOp::DeleteLoose}) {
            const auto empty = f.Run({.SourceId = id, .Op = op});
            expect(!empty.Changed);
            expect(f.Get(id).VertexCount() == 4u);
            expect(f.Get(id).FaceCount() == 1u);
        }
        const auto duplicated = f.Run({.SourceId = id, .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies, .CopyRotation = mat3{1.f}, .CopyTranslation = {4.f, 0.f, 0.f}, .Selection = selected});
        expect(duplicated.Changed);
        expect(duplicated.GeometryId == id);
        expect(duplicated.Created.Vertices.size() == 4u);
        expect(duplicated.Created.Faces.size() == 1u);
        expect(duplicated.Retained.Vertices.size() == 4u);
        expect(f.Get(id).FaceCount() == 2u);
        const auto &bounds = f.Store().GetSelectionRoot(id, Element::Vertex).Bounds;
        expect(bounds.Min.x == -1.f);
        expect(bounds.Max.x == 5.f);
        expect(f.Store().GetBoundaryEdges(id).Count() == 8u);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        expect(f.Store().Get(id).MeshletRoot == InvalidOffset);
        const auto extrude_id = f.Create(Quad());
        const auto extruded = f.Run({.SourceId = extrude_id, .Op = MeshTopologyOp::ExtrudeRegion, .Selection = All(f.Get(extrude_id))});
        expect(extruded.Changed);
        expect(extruded.Created.Vertices.size() == 4u);
        expect(f.Get(extrude_id).FaceCount() == 6u);
        expect(f.Get(extrude_id).HasClosedSurface());
        const auto extracted = f.Run({.SourceId = id, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = Element::Face, .Selection = GeometrySelection{.Faces = duplicated.Created.Faces}});
        expect(extracted.Changed);
        expect(extracted.GeometryId != id);
        expect(f.Get(extracted.GeometryId).VertexCount() == 4u);
        expect(f.Get(extracted.GeometryId).FaceCount() == 1u);
        expect(f.Store().GetBoundaryEdges(extracted.GeometryId).Count() == 4u);
        expect(f.Store().GetSelectionRoot(extracted.GeometryId, Element::Vertex).Bounds.Min.x == 3.f);
    };
    "headless loose deletion ignores document visibility and selection"_test = [] {
        GeometryFixture f;
        MeshData data{{{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {3.f, 0.f, 0.f}}};
        data.Edges.push_back({0u, 1u});
        const auto id = f.Create(std::move(data));
        const auto selected = All(f.Get(id));
        const std::array blocks{selected.Vertices.front() / MeshElementBlockSize};
        f.Store().EditHiddenBlocks(Element::Vertex, blocks, [](uint32_t, auto &words) { std::ranges::fill(words, ~0u); });
        const auto removed = f.Run({.SourceId = id, .Op = MeshTopologyOp::DeleteLoose, .Selection = selected});
        expect(removed.Changed);
        expect(f.Get(id).VertexCount() == 0u);
        expect(f.Get(id).EdgeCount() == 0u);
    };
    "extracted geometry chains edits and preserves source attributes"_test = [] {
        GeometryFixture f;
        MeshVertexAttributes attributes;
        const std::array texcoords{vec2{.2f, .7f}, vec2{.3f, .6f}, vec2{.4f, .5f}, vec2{.5f, .4f}};
        attributes.TexCoords0 = std::vector<vec2>(texcoords.begin(), texcoords.end());
        attributes.Colors0 = std::vector<vec4>(4u, {.1f, .3f, .5f, 1.f});
        attributes.Colors0ComponentCount = 4u;
        const auto source = CreateMesh(f.Scene, {.Data = Quad(), .Attrs = std::move(attributes)}).StoreId;
        const auto extracted = f.Run({.SourceId = source, .Op = MeshTopologyOp::KeepSelectedFaces, .SelectionElement = Element::Face, .Selection = All(f.Get(source))});
        const auto output_id = extracted.GeometryId;
        const auto selection = extracted.Created;
        expect(extracted.Changed);
        expect(output_id != source);
        expect(selection.Vertices.front() != **f.Get(source).vertices().begin());
        const auto original = f.Get(source).GetPosition(*f.Get(source).vertices().begin());
        const PositionOperationTarget target{.StoreId = output_id, .Selection = selection};
        const auto changes = ExecuteGeometryPositions(f.Scene, std::span{&target, 1u}, PositionEditOp::PushPull, .25f, 1u, {.Center = {0.f, 0.f, 2.f}});
        expect(changes.size() == 1u);
        expect(f.Get(source).GetPosition(*f.Get(source).vertices().begin()) == original);
        expect(f.Get(output_id).GetPosition(*f.Get(output_id).vertices().begin()).z > 0.f);
        {
            mtl::ComputeChain chain{f.Store().BufferContext()};
            const auto permuted = EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&target, 1u}, false, 0u, 0u);
            expect(permuted.size() == 1u);
            chain.Submit();
        }
        for (const auto corner : f.Get(source).fh_range(f.Get(source).FaceAt(0u)))
            expect(f.Store().Arenas().CornerUvs[0].Get(*corner) == texcoords[f.Get(source).VertexOrdinal(f.Get(source).GetToVertex(corner))]);
        for (const auto corner : f.Get(output_id).fh_range(f.Get(output_id).FaceAt(0u)))
            expect(f.Store().Arenas().CornerUvs[0].Get(*corner) == texcoords[(f.Get(output_id).VertexOrdinal(f.Get(output_id).GetToVertex(corner)) + 3u) % 4u]);
        const auto triangulated = f.Run({.SourceId = output_id, .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = selection});
        expect(triangulated.Changed);
        expect(triangulated.Created.Faces.size() == 1u);
        expect(triangulated.Retained.Faces.size() == 1u);
        expect(f.Get(output_id).FaceCount() == 2u);
        expect(f.Get(source).FaceCount() == 1u);
        for (const auto face : f.Get(output_id).faces())
            for (const auto corner : f.Get(output_id).fh_range(face)) {
                expect(f.Store().Arenas().CornerUvs[0].Get(*corner) == texcoords[(f.Get(output_id).VertexOrdinal(f.Get(output_id).GetToVertex(corner)) + 3u) % 4u]);
                expect(f.Store().Arenas().CornerColors.Get(*corner) == vec4{.1f, .3f, .5f, 1.f});
            }
        expect(f.Store().Get(output_id).MeshletRoot == InvalidOffset);
        expect(f.Scene.Living.size() == 0u);
    };
    "headless affine transforms share editor frame math"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selection = All(f.Get(id));
        const auto first = selection.Vertices.front();
        const auto untouched = f.Get(id).GetPosition(he::VH{selection.Vertices.back()});
        const quat quarter_turn{.7071067811865475f, 0.f, 0.f, .7071067811865475f};
        const PositionOperationTarget target{.StoreId = id, .Selection = GeometrySelection{.Vertices = {first}}, .World = {.P = {10.f, 20.f, 3.f}, .R = quarter_turn, .S = {2.f, 3.f, 4.f}}};
        const PositionOperationOptions options{.Transform = PositionTransform{.Delta = {.P = {1.f, 2.f, 3.f}, .R = quarter_turn, .S = {2.f, .5f, 1.f}}, .Pivot = {10.f, 20.f, 3.f}}};
        const auto changed = ExecuteGeometryPositions(f.Scene, std::span{&target, 1u}, PositionEditOp::Transform, 1.f, 1u, options);
        expect(changed.size() == 1u);
        const auto position = f.Get(id).GetPosition(he::VH{first});
        expect(std::abs(position.x - 4.f) < 1e-5f);
        expect(std::abs(position.y + 2.f / 3.f) < 1e-5f);
        expect(std::abs(position.z - .75f) < 1e-5f);
        expect(f.Get(id).GetPosition(he::VH{selection.Vertices.back()}) == untouched);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        const auto triangulated = f.Run({.SourceId = id, .Op = MeshTopologyOp::Triangulate, .SelectionElement = Element::Face, .Selection = selection});
        expect(triangulated.Changed);
        expect(f.Get(id).FaceCount() == 2u);
        expect(f.Store().Get(id).MeshletRoot == InvalidOffset);
    };
    "geometry planners reject foreign masks and invalid planes before reading topology"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad()), other = f.Create(Quad());
        const auto mesh = f.Get(id);
        const auto selected = All(mesh), foreign = All(f.Get(other));
        const auto rejects = [](auto &&plan) {
            try {
                plan();
            } catch (const std::invalid_argument &) { return true; }
            return false;
        };
        expect(rejects([&] { FillTask(f.Store(), mesh, foreign); }));
        expect(rejects([&] { BeautifyFaceTask(mesh, foreign, false); }));
        expect(rejects([&] { UnsubdivideTask(mesh, foreign); }));
        expect(rejects([&] { DecimateTask(f.Store(), mesh, selected, .5f, foreign.Faces); }));
        expect(rejects([&] { BisectTasks(mesh, {}, {}, false, false); }));
        expect(rejects([&] { SymmetrizeTasks(mesh, 3u, false); }));
        expect(rejects([&] { KnifeTask(mesh, selected, mat4{1.f}, {}, {}, {}); }));
        expect(rejects([&] { MergeTask(f.Store(), mesh, selected, GeometryMergeMode::First, 0.f, foreign.Vertices.front()); }));
        expect(mesh.FaceCount() == 1u);
    };

    "headless sharpness uses explicit masks and refreshes canonical normals"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto all = All(f.Get(id));
        mtl::ComputeChain chain{f.Store().BufferContext()};
        const auto apply = [&](EditSharpnessOperation operation, GeometrySelection selection = {}, bool value = true) {
            const SharpnessOperationTarget target{id, std::move(selection)};
            return ExecuteGeometrySharpness(f.Scene, chain, std::span{&target, 1u}, operation, value, .1f);
        };
        expect(apply(EditSharpnessOperation::SetSelectedFaces).empty());
        expect(apply(EditSharpnessOperation::SetSelectedFaces, {.Faces = all.Faces}).size() == 1u);
        expect(f.Store().Arenas().FaceSharpness.Get({all.Faces.front(), 1u})[0] == 1u);
        // A local sharpness edit may retain mixed classification without scanning the mesh.
        expect(f.Store().GetCornerClassMode(id) != uint32_t(CornerClassMode::UniformVertex));
        expect((f.Store().GetSelectionRoot(id, Element::Face).Flags & SelectionLiveSharp) != 0u);
        apply(EditSharpnessOperation::SmoothAll);
        apply(EditSharpnessOperation::SetSelectedEdges, {.Edges = {all.Edges.front()}});
        expect(f.Store().Arenas().EdgeSharpness.Get({all.Edges.front(), 1u})[0] == 1u);
        apply(EditSharpnessOperation::SmoothAll);
        apply(EditSharpnessOperation::SetVertexEdges, {.Vertices = {all.Vertices[0], all.Vertices[1]}});
        uint32_t sharp = 0u;
        for (const auto edge : all.Edges) sharp += f.Store().Arenas().EdgeSharpness.Get({edge, 1u})[0] != 0u;
        expect(sharp == 3u);
        apply(EditSharpnessOperation::SetAllFaces);
        expect(f.Store().Arenas().FaceSharpness.Get({all.Faces.front(), 1u})[0] == 1u);
        apply(EditSharpnessOperation::SmoothByAngle);
        for (const auto edge : all.Edges) expect(f.Store().Arenas().EdgeSharpness.Get({edge, 1u})[0] == 0u);
        expect(f.Get(id).GetNormal(he::FH{all.Faces.front()}).z > .99f);
        expect(f.Store().GetSelectedElements(id, Element::Vertex).Count() == 0u);
        expect(f.Store().Get(id).MeshletRoot == InvalidOffset);
        expect(f.Scene.Living.empty());
    };
    "headless knife restricts canonical candidates without render caches"_test = [] {
        GeometryFixture f;
        const auto id = f.Create(Quad());
        const auto selected = All(f.Get(id));
        const auto cut = f.Run({.SourceId = id, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagScreenCuts, .ScreenTransform = mat4{1.f}, .Extent = {100.f, 100.f}, .KnifeStart = {50.f, 0.f}, .KnifeEnd = {50.f, 100.f}, .SelectionElement = Element::Face, .Selection = GeometrySelection{.Faces = selected.Faces}});
        expect(cut.Changed);
        expect(f.Get(id).VertexCount() == 6u);
        expect(f.Get(id).FaceCount() == 2u);
        expect(cut.Created.Vertices.size() == 2u);
        expect(f.Store().Get(id).MeshletRoot == InvalidOffset);
        MeshData separated;
        for (uint32_t face = 0u; face < 512u; ++face) {
            const auto first = uint32_t(separated.Positions.size());
            const float x = face < 256u ? 0.f : 100.f;
            separated.Positions.insert(separated.Positions.end(), {{x - 1.f, -1.f, 0.f}, {x + 1.f, -1.f, 0.f}, {x + 1.f, 1.f, 0.f}, {x - 1.f, 1.f, 0.f}});
            separated.AddFace(std::array{first, first + 1u, first + 2u, first + 3u});
        }
        const auto separated_id = f.Create(std::move(separated));
        for (const auto mode : {TopologyFlagPlaneCuts, TopologyFlagScreenCuts}) {
            const MeshTopologyTask query{.SourceId = separated_id, .Op = MeshTopologyOp::Subdivide, .Flags = mode | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}, .ScreenTransform = mat4{1.f}, .Extent = {100.f, 100.f}, .KnifeStart = {50.f, 0.f}, .KnifeEnd = {50.f, 100.f}};
            mtl::ComputeChain chain{f.Buffers};
            SpatialFaceWork work{f.Scene, chain, query};
            chain.Submit();
            expect(work.CandidateCount == 256u);
            expect(work.CandidateBlocks == 1u);
            work.RecordFaces(f.Scene, chain);
            chain.Submit();
            expect(work.Count == 256u);
        }
        auto moved = All(f.Get(separated_id));
        moved.Vertices.resize(1024u);
        const PositionOperationTarget moving{.StoreId = separated_id, .Selection = GeometrySelection{.Vertices = moved.Vertices}};
        ExecuteGeometryPositions(f.Scene, std::span{&moving, 1u}, PositionEditOp::Transform, 1.f, 1u, {.Transform = PositionTransform{.Delta = {.P = {200.f, 0.f, 0.f}}}});
        const MeshTopologyTask query{.SourceId = separated_id, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}};
        mtl::ComputeChain chain{f.Buffers};
        SpatialFaceWork moved_work{f.Scene, chain, query};
        chain.Submit();
        expect(moved_work.CandidateCount == 0u);
        expect(f.Store().GetSelectionRoot(separated_id, Element::Face).Bounds.Min.x == 99.f);
        for (const auto input : {id, f.Create(Quad())}) {
            const auto before = f.Get(input).FaceCount();
            const auto stages = BisectTasks(f.Get(input), vec3{}, vec3{1.f, 0.f, 0.f}, true, false);
            expect(throws<std::invalid_argument>([&] { ExecuteGeometryTopology(f.Scene, stages); }));
            expect(f.Get(input).FaceCount() == before);
            const auto staged = ExecuteGeometryTopologyStages(f.Scene, stages);
            expect(staged.size() == stages.size());
            expect(staged.back().Changed);
            expect(f.Get(input).FaceCount() == 1u);
            for (const auto v : f.Get(input).vertices()) expect(f.Get(input).GetPosition(v).x >= 0.f);
        }
        const auto through_vertex = f.Create(MeshData{{{-1.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, {1.f, -1.f, 0.f}}, {{0u, 1u, 2u}}});
        const MeshTopologyTask vertex_query{.SourceId = through_vertex, .Op = MeshTopologyOp::Subdivide, .Flags = TopologyFlagPlaneCuts | TopologyFlagSelectAll, .PlaneNormal = {1.f, 0.f, 0.f}};
        SpatialFaceWork vertex_work{f.Scene, chain, vertex_query};
        chain.Submit();
        vertex_work.RecordFaces(f.Scene, chain);
        chain.Submit();
        expect(vertex_work.Count == 1u);
    };
    "topology operators read mask selections like handle lists"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        for (const auto &c : TopologyCases) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto listed = PlanCase(c, f.Store(), f.Get(listed_id));
                auto masked = PlanCase(c, f.Store(), f.Get(masked_id));
                expect(fatal(listed.has_value() && masked.has_value()));
                masked->Selection = masks.Write(f.Store(), std::get<GeometrySelection>(masked->Selection));
                const auto by_list = Describe(f.Store(), listed_id, f.Run(*listed));
                const auto by_masks = Describe(f.Store(), masked_id, f.Run(*masked));
                expect(by_list.Changed);
                expect(by_list == by_masks);
            };
        }
    };
    "topology writes emitted selection into explicit destination masks"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        const auto id = f.Create(Quad());
        const auto output = masks.Write(f.Store(), {});
        const auto duplicated = f.Run({.SourceId = id, .Op = MeshTopologyOp::DuplicateGeometry, .Flags = TopologyFlagTransformCopies, .CopyRotation = mat3{1.f}, .CopyTranslation = {4.f, 0.f, 0.f}, .Selection = All(f.Get(id)), .SelectionOutput = output});
        expect(duplicated.Changed);
        for (const auto element : {Element::Vertex, Element::Edge, Element::Face})
            expect(ReadMasks(f.Store(), id, element, output.Get(element)) == duplicated.Created.Get(element));
        // The store's own masks receive nothing.
        for (const auto element : {Element::Vertex, Element::Edge, Element::Face})
            expect(f.Store().GetSelectedElements(id, element).Count() == 0u);
    };
    "position and sharpness operations read mask selections like handle lists"_test = [] {
        GeometryFixture f;
        MaskWriter masks{f.Buffers};
        struct PositionCase {
            std::string_view Name;
            PositionEditOp Op;
            float Factor;
            PositionOperationOptions Options;
        };
        const std::array<PositionCase, 6> positions{{
            {"Smooth", PositionEditOp::Smooth, .5f, {}},
            {"PushPull", PositionEditOp::PushPull, .25f, {.Center = {0.f, 0.f, 2.f}}},
            {"Flatten", PositionEditOp::Copy, 1.f, {}},
            {"Circularize", PositionEditOp::Copy, 1.f, {.Circle = PositionCircle{}}},
            {"ShrinkFatten", PositionEditOp::ShrinkFatten, .1f, {.Flags = PositionEditSelectedFaceNormals}},
            {"VertexSlide", PositionEditOp::VertexSlide, .5f, {.Slide = PositionSlide{}}},
        }};
        // Exclusions name the triangulated cell's faces.
        const auto excluded = [](const Mesh &mesh) {
            GeometrySelection result;
            for (const auto face : mesh.faces())
                if (mesh.GetValence(face) == 3u) result.Faces.push_back(*face);
            return result;
        };
        for (const auto &c : positions) {
            boost::ut::test(std::string{c.Name}) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const auto listed_selection = Region(f.Get(listed_id)), masked_selection = Region(f.Get(masked_id));
                const PositionOperationTarget listed{.StoreId = listed_id, .Selection = listed_selection, .Excluded = excluded(f.Get(listed_id)), .Reference = listed_selection.Vertices[4]};
                const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), masked_selection), .Excluded = masks.Write(f.Store(), excluded(f.Get(masked_id))), .Reference = masked_selection.Vertices[4]};
                const auto before = Points(f.Get(listed_id));
                const auto listed_changes = ExecuteGeometryPositions(f.Scene, std::span{&listed, 1u}, c.Op, c.Factor, 1u, c.Options);
                const auto masked_changes = ExecuteGeometryPositions(f.Scene, std::span{&masked, 1u}, c.Op, c.Factor, 1u, c.Options);
                expect(listed_changes.size() == masked_changes.size());
                expect(!std::ranges::equal(Points(f.Get(listed_id)), before));
                expect(std::ranges::equal(Points(f.Get(listed_id)), Points(f.Get(masked_id))));
            };
        }
        "face attributes"_test = [&] {
            const auto uvs = [] {
                MeshVertexAttributes attributes;
                std::vector<vec2> texcoords;
                for (uint32_t i = 0u; i < 28u; ++i) texcoords.push_back({float(i) / 28.f, 1.f - float(i) / 28.f});
                attributes.TexCoords0 = std::move(texcoords);
                return attributes;
            };
            const auto listed_id = CreateMesh(f.Scene, {.Data = Patchwork(), .Attrs = uvs()}).StoreId, masked_id = CreateMesh(f.Scene, {.Data = Patchwork(), .Attrs = uvs()}).StoreId;
            const PositionOperationTarget listed{.StoreId = listed_id, .Selection = Region(f.Get(listed_id))};
            const PositionOperationTarget masked{.StoreId = masked_id, .Selection = masks.Write(f.Store(), Region(f.Get(masked_id)))};
            mtl::ComputeChain chain{f.Store().BufferContext()};
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&listed, 1u}, false, 0u, 1u).size() == 1u);
            expect(EncodeFaceAttributeOperation(f.Store(), GetMeshPipelines(f.Scene), chain, std::span{&masked, 1u}, false, 0u, 1u).size() == 1u);
            chain.Submit();
            const auto corner_uvs = [&](uint32_t id) {
                std::vector<vec2> result;
                for (const auto face : f.Get(id).faces())
                    for (const auto corner : f.Get(id).fh_range(face)) result.push_back(f.Store().Arenas().CornerUvs[0].Get(*corner));
                return result;
            };
            expect(std::ranges::equal(corner_uvs(listed_id), corner_uvs(masked_id)));
        };
        for (const auto operation : {EditSharpnessOperation::SetSelectedFaces, EditSharpnessOperation::SetSelectedEdges, EditSharpnessOperation::SetVertexEdges, EditSharpnessOperation::SetAllFaces, EditSharpnessOperation::SmoothByAngle}) {
            boost::ut::test("sharpness " + std::to_string(uint32_t(operation))) = [&] {
                const auto listed_id = f.Create(Patchwork()), masked_id = f.Create(Patchwork());
                const SharpnessOperationTarget listed{listed_id, Region(f.Get(listed_id))};
                const SharpnessOperationTarget masked{masked_id, masks.Write(f.Store(), Region(f.Get(masked_id)))};
                mtl::ComputeChain chain{f.Store().BufferContext()};
                const auto listed_changes = ExecuteGeometrySharpness(f.Scene, chain, std::span{&listed, 1u}, operation, true, .1f);
                const auto masked_changes = ExecuteGeometrySharpness(f.Scene, chain, std::span{&masked, 1u}, operation, true, .1f);
                expect(listed_changes.size() == 1u);
                expect(masked_changes.size() == 1u);
                const auto sharpness = [&](uint32_t id) {
                    std::vector<uint8_t> result;
                    for (const auto face : f.Get(id).faces()) result.push_back(f.Store().Arenas().FaceSharpness.Get({*face, 1u})[0]);
                    for (const auto edge : f.Get(id).edges()) result.push_back(f.Store().Arenas().EdgeSharpness.Get({*edge, 1u})[0]);
                    return result;
                };
                expect(std::ranges::equal(sharpness(listed_id), sharpness(masked_id)));
            };
        }
    };
}
