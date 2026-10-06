// Exercises mesh edits through production actions and viewport rendering.
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "gpu/PBRMaterial.h"
#include "mesh/Mesh.h"
#include "mesh/MeshClone.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshConnectivityGpu.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshEdgeUsers.h"
#include "mesh/MeshPipelines.h"
#include "metal/Dispatch.h"
#include "numeric/MatrixMath.h"
#include "numeric/QuaternionMath.h"
#include "object/ObjectOps.h"
#include "project/Project.h"
#include "render/ElementWorkOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MeshletBuildGpu.h"
#include "render/SceneUpdates.h"
#include "scene/Entity.h"
#include "selection/SelectionGpu.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <numbers>
#include <set>
#include <string>
#include <utility>
#include <vector>

using boost::ut::expect;

namespace {
struct Counts {
    uint32_t Vertices, Edges, Faces, Triangles;
    bool operator==(const Counts &) const = default;
};

Counts CountsOf(const Mesh &mesh) {
    return {mesh.VertexCount(), mesh.EdgeCount(), mesh.FaceCount(), mesh.TriangleIndexCount() / 3u};
}

void ExpectCounts(const Mesh &mesh, Counts expected) {
    const auto actual = CountsOf(mesh);
    expect(actual == expected) << "counts" << actual.Vertices << actual.Edges << actual.Faces << actual.Triangles;
}

bool Near(vec3 actual, vec3 expected) {
    return Length(actual - expected) < 1e-5f;
}

using VertexPositions = std::vector<std::pair<he::VH, vec3>>;

VertexPositions Positions(const Mesh &mesh) {
    VertexPositions positions;
    for (const auto vertex : mesh.vertices()) positions.emplace_back(vertex, mesh.GetPosition(vertex));
    return positions;
}

std::vector<vec3> FacePositions(const Mesh &mesh, he::FH face) {
    std::vector<vec3> positions;
    for (const auto vertex : mesh.fv_range(face)) positions.push_back(mesh.GetPosition(vertex));
    return positions;
}

void ExpectPositions(const Mesh &mesh, const VertexPositions &expected) {
    expect(mesh.VertexCount() == expected.size());
    for (const auto &[vertex, position] : expected) {
        const auto actual = mesh.GetPosition(vertex);
        expect(Near(actual, position)) << "vertex" << *vertex << "actual" << actual.x << actual.y << actual.z
                                       << "expected" << position.x << position.y << position.z;
    }
}

// Check the sparse summaries against canonical membership, masks and positions.
void ExpectSelectionIndex(const MeshStore &meshes, uint32_t id) {
    if (!meshes.Get(id).SelectionSummary.Count) return;
    const Mesh mesh{meshes, id};
    const std::array elements{Element::Vertex, Element::Edge, Element::Face};
    const std::array domains{MeshStore::ElementDomain::Vertex, MeshStore::ElementDomain::Edge, MeshStore::ElementDomain::Face};
    for (uint32_t d = 0u; d < elements.size(); ++d) {
        const auto selection = meshes.GetSelectedElements(id, elements[d]);
        const auto hidden = meshes.GetHiddenElements(id, elements[d]);
        std::vector<uint32_t> expected_hidden, actual_hidden;
        std::vector<uint32_t> expected, actual;
        uint32_t live_count = 0u;
        vec3 sum{};
        AABB bounds;
        WithDomain(meshes.Arenas(), domains[d], [&](const auto &arena) {
            arena.ForEach(DomainSet(meshes.Get(id), domains[d]), [&](uint32_t v, uint32_t) {
                ++live_count;
                const bool selected = selection.Bits[v / 32u] & (1u << (v % 32u));
                if (selected) expected.push_back(v);
                if (hidden.Contains(v)) {
                    expected_hidden.push_back(v);
                    expect(!selected);
                }
                if (d == 0u) {
                    const auto p = mesh.GetPosition(he::VH{v});
                    bounds.Min = Min(bounds.Min, p);
                    bounds.Max = Max(bounds.Max, p);
                    if (selected) sum += p;
                }
            });
        });
        std::ranges::sort(expected);
        selection.ForEach([&](uint32_t v) { actual.push_back(v); });
        expect(actual == expected);
        expect(selection.Count() == expected.size());
        expect(selection.First() == (expected.empty() ? std::optional<uint32_t>{} : expected.front()));
        expect(selection.Last() == (expected.empty() ? std::optional<uint32_t>{} : expected.back()));
        const auto &root = meshes.GetSelectionRoot(id, elements[d]);
        expect(root.LiveCount == live_count);
        std::ranges::sort(expected_hidden);
        hidden.ForEach([&](uint32_t h) { actual_hidden.push_back(h); });
        expect(actual_hidden == expected_hidden);
        expect(root.Hidden == expected_hidden.size());
        expect(hidden.Count() == expected_hidden.size());
        expect(Length(root.PositionSum - sum) < 0.0001f * (1.f + Length(sum)));
        expect(root.Bounds == bounds);
    }
    std::vector<uint32_t> expected_boundary, actual_boundary;
    for (const auto e : mesh.edges())
        if (!mesh.GetOppositeHalfedge(mesh.GetHalfedge(e, 0u))) expected_boundary.push_back(*e);
    std::ranges::sort(expected_boundary);
    meshes.GetBoundaryEdges(id).ForEach([&](uint32_t e) { actual_boundary.push_back(e); });
    expect(expected_boundary == actual_boundary);
}

// Each checkpoint submits a real viewport frame and checks the action error list.
struct Fixture : Engine {
    TestDir Dir;

    explicit Fixture(const char *name)
        : Engine{false}, Dir{(std::string{"/tmp/mesheditor-scratch/topology-"} + name).c_str()} {
        expect(P->Begin(Dir));
        R.Context.get<ViewportExtent>().Value = {64u, 64u};
        P->Settle();
    }

    template<typename A> void Do(A a) {
        P->Do(action::MakeAction(std::move(a)));
    }

    template<typename A> void Stage(A a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
    }

    void Commit() {
        action::Commit();
        P->Frame(action::Drain());
        expect(!P->HasStaged());
    }

    void Cancel() {
        P->CancelGesture();
        P->Settle();
        expect(!P->HasStaged());
    }

    void Checkpoint() {
        SubmitViewport(R, Viewport);
        WaitForRender(R);
        expect(ViewportImageReady(R));
        for (const auto &message : R.Context.get<action::Errors>().Messages)
            std::printf("action error: %s\n", message.c_str());
        expect(R.Context.get<action::Errors>().Messages.empty());
    }

    void CheckUndoRedo(auto &&check) {
        P->Undo();
        Checkpoint();
        check(R, false);
        P->Redo();
        Checkpoint();
        check(R, true);
    }

    void CheckSaved(auto &&check) {
        expect(P->Save());
        expect(P->Close());
        Engine reopened{false};
        expect(reopened.P->Open(Dir));
        reopened.P->Settle();
        check(reopened.R, true);
    }

    void CheckSavedMesh(state::Entity entity) {
        const auto mesh = GetMesh(R, entity);
        const auto counts = CountsOf(mesh);
        const auto positions = Positions(mesh);
        CheckSaved([&](const state::Scene &scene, bool) {
            const auto restored = GetMesh(scene, entity);
            ExpectCounts(restored, counts);
            ExpectPositions(restored, positions);
        });
    }

    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
    }

    auto AddEditable(uint32_t id, Element element, MeshInstanceCreateInfo info = {}) {
        const auto added = AddMesh(R, id, info);
        P->Settle();
        Do(action::selection::Select{added.second});
        EnterEdit(element);
        return added;
    }

    void SelectElements(state::Entity entity, std::span<const uint32_t> ordinals, Element element) {
        ApplyEditSelectionLists(R, std::array{std::pair{entity, ordinals}}, element);
        P->Settle();
    }

    void Cube(Element element) {
        Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
        EnterEdit(element);
    }

    Mesh ActiveMesh() const {
        return GetMesh(R, GetActiveMeshEntity(R));
    }

    Mesh MeshOf(uint32_t id) const {
        return Mesh{R.Context.get<const MeshStore>(), id};
    }

    auto Selection(Element element) const {
        return R.Context.get<const MeshStore>().GetSelectedElements(ActiveMesh().GetStoreId(), element);
    }

    std::unique_ptr<RenderView> View() const {
        return std::make_unique<RenderView>(R.Context.get<const GpuBuffers>().FrameView);
    }

    void Click(vec2 at, bool toggle = false) {
        Do(action::selection::ApplyEditElementClick{at, toggle, View()});
    }

    const SceneViewUBO &FrameView() const {
        const auto bytes = R.Context.get<const GpuBuffers>().SceneViewUBO.Contents();
        return *reinterpret_cast<const SceneViewUBO *>(bytes.data());
    }

    vec2 ViewFraction(vec3 position) const {
        const auto clip = FrameView().ViewProj * vec4{position.x, position.y, position.z, 1.f};
        return {0.5f + 0.5f * clip.x / clip.w, 0.5f - 0.5f * clip.y / clip.w};
    }
};

// The small mesh's finest render clusters must contain the expected kind and primitive count.
void ExpectRenderTopology(const Fixture &f, uint32_t id, uint32_t topology, uint32_t primitives) {
    const auto &meshes = f.R.Context.get<const MeshStore>();
    const auto &owner = meshes.Get(id);
    expect(owner.RenderTopologies == (1u << topology));
    expect(meshes.MeshletCount(owner) > 0u);
    uint32_t actual = 0u;
    meshes.Render().ActiveMeshlets.ForEach(owner.MeshletRoot, [&](uint32_t handle) {
        const auto &meshlet = meshes.Render().Meshlets.Get({handle, 1u})[0];
        expect(meshlet.Topology == topology);
        actual += meshlet.TriangleCount;
    });
    expect(actual == primitives);
}

std::array<uint32_t, 3> RenderedPrimitiveCounts(const MeshStore &meshes, uint32_t id) {
    std::array<uint32_t, 3> counts{};
    meshes.Render().ActiveMeshlets.ForEach(meshes.Get(id).MeshletRoot, [&](uint32_t handle) {
        const auto &cluster = meshes.Render().Meshlets.Get({handle, 1u})[0];
        counts[cluster.Topology] += cluster.TriangleCount;
    });
    return counts;
}

void ExpectRenderedGeometry(const MeshStore &meshes, const Mesh &mesh) {
    std::array<uint32_t, 3> expected{mesh.TriangleIndexCount() / 3u, 0u, 0u};
    for (const auto edge : mesh.edges()) expected[1] += !mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(edge, 0u));
    for (const auto vertex : mesh.vertices()) expected[2] += mesh.GetConnectivity().VertexCorners[*vertex].y == 0u;
    expect(RenderedPrimitiveCounts(meshes, mesh.GetStoreId()) == expected);
}

// Preserve explicit corner layers and loose edges without source welding.
uint32_t CreateMixedFixture(state::Scene &scene, const MeshData &data, std::span<const std::array<uint32_t, 2>> wires, bool attributes) {
    auto &meshes = scene.Context.get<MeshStore>();
    const auto &a = meshes.Arenas();
    const auto id = meshes.CreateMeshSource(MeshData{data.Positions});
    const auto count = uint32_t(data.FaceCorners.size() + 2u * wires.size());
    meshes.AllocateConnectivity(id, count, data.FaceCount(), !data.FaceOffsets.empty(), data.FaceOffsets, wires);
    auto corners = a.FaceCorners.Buffer.GetMutableSpan<uint32_t>(a.FaceCorners.Dense(meshes.Get(id).FaceCorners));
    const auto first = a.Vertices.First(meshes.Get(id).Vertices);
    for (uint32_t c = 0u; c < data.FaceCorners.size(); ++c) corners[c] = first + data.FaceCorners[c];
    BuildConnectivityNow(scene, std::array{id});
    CornerLayers layers;
    if (attributes) {
        layers.Colors.assign(count, vec4{.2f, .4f, .6f, 1.f});
        layers.Uvs[0].assign(count, vec2{.3f, .7f});
    }
    meshes.CreateMesh(id, data, {}, {}, layers, false);
    meshes.WriteRecord(id).Classification = uint32_t(CornerClassMode::UniformFace);
    return id;
}

void TestCubeSelectionMove() {
    Fixture f{"cube-selection-move"};
    f.Cube(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 12u, 6u, 12u});
    expect(f.Selection(Element::Vertex).Count() == 8u);

    const auto original = Positions(f.ActiveMesh());
    const auto eye = f.FrameView().CameraPosition;
    const auto nearest = std::ranges::max_element(original, {}, [&](const auto &item) {
        return Dot(item.second, eye);
    });
    expect(nearest != original.end());
    if (nearest == original.end()) return;
    const auto picked = nearest->first;
    const auto at = f.ViewFraction(nearest->second);

    // The first click activates the selected vertex and the second removes it.
    f.Click(at, true);
    expect(f.Selection(Element::Vertex).Count() == 8u);
    f.Click(at, true);
    expect(f.Selection(Element::Vertex).Count() == 7u);
    expect(!f.Selection(Element::Vertex).Contains(*picked));
    f.Checkpoint();

    const vec3 Offset{0.25f, -0.5f, 0.75f};
    f.Stage(action::view::TransformElements{{.P = Offset}});
    expect(f.P->HasStaged());
    f.Commit();
    f.Checkpoint();
    const auto moved = f.ActiveMesh();
    ExpectCounts(moved, {8u, 12u, 6u, 12u});
    expect(f.Selection(Element::Vertex).Count() == 7u);
    for (const auto &[vertex, position] : original) {
        const auto expected = vertex == picked ? position : position + Offset;
        expect(Near(moved.GetPosition(vertex), expected));
    }
    ExpectRenderTopology(f, moved.GetStoreId(), 0u, 12u);
}

void TestInsetGesture() {
    Fixture f{"inset-gesture"};
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto instance = FindActiveEntity(f.R);
    const auto entity = GetActiveMeshEntity(f.R);
    const auto id = f.R.get<const MeshHandle>(entity).StoreId;

    auto other_info = std::make_unique<MeshInstanceCreateInfo>();
    other_info->Transform.P = {5.f, 0.f, 0.f};
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::move(other_info)});
    const auto other_id = f.ActiveMesh().GetStoreId();
    const auto other_positions = Positions(f.MeshOf(other_id));
    f.Do(action::selection::Select{instance});
    f.EnterEdit(Element::Face);
    f.Checkpoint();
    f.Click({0.5f, 0.5f});
    expect(f.Selection(Element::Face).Count() == 1u);

    he::FH picked;
    f.Selection(Element::Face).ForEach([&](uint32_t face) { picked = he::FH{face}; });
    expect(bool(picked));
    if (!picked) return;
    const auto original = Positions(f.ActiveMesh());
    const auto face_positions = FacePositions(f.ActiveMesh(), picked);
    const auto center = f.ActiveMesh().CalcFaceCentroid(picked);
    expect(face_positions.size() == 4u);

    he::FH untouched;
    for (const auto face : f.ActiveMesh().faces()) {
        if (face != picked) {
            untouched = face;
            break;
        }
    }
    const auto untouched_positions = FacePositions(f.ActiveMesh(), untouched);
    const auto check_unchanged = [&] {
        const auto mesh = f.ActiveMesh();
        for (const auto &[vertex, position] : original) expect(Near(mesh.GetPosition(vertex), position));
        expect(FacePositions(mesh, untouched) == untouched_positions);
        ExpectCounts(f.MeshOf(other_id), {8u, 12u, 6u, 12u});
        ExpectPositions(f.MeshOf(other_id), other_positions);
        expect(f.R.get<const MeshHandle>(entity).StoreId == id);
    };
    const auto check_inset = [&](float scale) {
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {12u, 20u, 10u, 20u});
        expect(f.Selection(Element::Face).Count() == 1u);
        he::FH inner;
        f.Selection(Element::Face).ForEach([&](uint32_t face) { inner = he::FH{face}; });
        expect(bool(inner));
        if (!inner) return;
        const auto actual = FacePositions(mesh, inner);
        expect(actual.size() == 4u);
        for (const auto position : face_positions) {
            const auto expected = center + (position - center) * scale;
            expect(std::ranges::any_of(actual, [&](vec3 value) { return Near(value, expected); }));
        }
        check_unchanged();
    };

    f.Stage(action::mesh::Inset{.Thickness = 0.1f});
    expect(f.P->HasStaged());
    f.Checkpoint();
    check_inset(0.9f);

    // A second update replaces the preview from the same original face.
    f.Stage(action::mesh::Inset{.Thickness = 0.25f});
    f.Checkpoint();
    check_inset(0.75f);
    f.Commit();
    f.Checkpoint();
    check_inset(0.75f);
    const auto committed = Positions(f.ActiveMesh());
    ExpectRenderTopology(f, id, 0u, 20u);

    f.P->Undo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 12u, 6u, 12u});
    ExpectPositions(f.ActiveMesh(), original);
    check_unchanged();
    f.P->Redo();
    f.Checkpoint();
    check_inset(0.75f);

    f.Stage(action::mesh::Inset{.Thickness = 0.1f});
    expect(f.P->HasStaged());
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {16u, 28u, 14u, 28u});
    f.Cancel();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {12u, 20u, 10u, 20u});
    ExpectPositions(f.ActiveMesh(), committed);
    check_inset(0.75f);
    ExpectRenderTopology(f, id, 0u, 20u);
    ExpectRenderTopology(f, other_id, 0u, 12u);
    f.CheckSavedMesh(entity);
}

void TestPointHullDelete() {
    Fixture f{"point-hull-delete"};
    MeshSource source;
    source.Data.Positions = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 0u, 0u, 0u});
    ExpectRenderTopology(f, id, 2u, 4u);

    f.Do(action::mesh::ConvexHull{});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 6u, 4u, 4u});
    ExpectPositions(f.ActiveMesh(), original);
    expect(f.R.get<const MeshHandle>(entity).StoreId == id);
    ExpectRenderTopology(f, id, 0u, 4u);

    f.Do(action::view::SetEditMode{.Mode = Element::Face});
    f.Do(action::selection::SelectAll{});
    expect(f.Selection(Element::Face).Count() == 4u);
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 6u, 0u, 0u});
    ExpectPositions(f.ActiveMesh(), original);
    ExpectRenderTopology(f, id, 1u, 6u);

    f.P->Undo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 6u, 4u, 4u});
    ExpectPositions(f.ActiveMesh(), original);
    ExpectRenderTopology(f, id, 0u, 4u);
}

void TestOnlyFacesPreservesEdges() {
    for (const bool nonmanifold : {false, true}) {
        Fixture f{"only-faces-preserves-edges"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {4, 0, 0}, {5, 0, 0}, {6, 1, 0}};
        if (nonmanifold) {
            data.AddFace(std::array{0u, 1u, 2u});
            data.AddFace(std::array{1u, 0u, 3u});
            data.AddFace(std::array{0u, 1u, 4u});
        } else data.AddFace(std::array{0u, 1u, 2u, 3u});
        const std::array<std::array<uint32_t, 2>, 2> wires{{{5u, 6u}, {0u, 1u}}};
        const auto id = CreateMixedFixture(f.R, data, wires, false);
        const auto [entity, instance] = f.AddEditable(id, Element::Face);
        f.SelectElements(entity, nonmanifold ? std::vector<uint32_t>{0u, 1u} : std::vector<uint32_t>{0u}, Element::Face);
        const auto original = f.ActiveMesh();
        const auto positions = Positions(original);
        std::map<he::EH, uint64_t> edges;
        for (const auto e : original.edges()) {
            const auto h = original.GetHalfedge(e, 0u);
            edges.emplace(e, MeshEdgeUsers::Key(*original.GetFromVertex(h), *original.GetToVertex(h)));
        }
        f.Do(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {8u, nonmanifold ? 9u : 6u, uint32_t(nonmanifold), uint32_t(nonmanifold)});
        ExpectPositions(mesh, positions);
        for (const auto &[e, key] : edges) {
            const auto h = mesh.GetHalfedge(e, 0u);
            expect(MeshEdgeUsers::Key(*mesh.GetFromVertex(h), *mesh.GetToVertex(h)) == key);
            if (!mesh.GetConnectivity().FaceOf(h)) expect(mesh.GetOppositeHalfedge(mesh.GetOppositeHalfedge(h)) == h);
        }
        expect(RenderedPrimitiveCounts(f.R.Context.get<const MeshStore>(), id)[1] == 6u);
    }
}

// A surface, a coincident wire, an attached wire, a separate wire, and a point.
uint32_t MixedDeleteFixture(Fixture &f) {
    MeshData data;
    data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 1, 0}, {4, 0, 0}, {5, 0, 0}, {6, 1, 0}};
    data.AddFace(std::array{0u, 1u, 2u, 3u});
    const std::array<std::array<uint32_t, 2>, 3> wires{{{0u, 1u}, {2u, 4u}, {5u, 6u}}};
    return CreateMixedFixture(f.R, data, wires, true);
}

void TestDeleteEdgesPreservesOtherEdges() {
    for (const bool keep_vertices : {false, true}) {
        Fixture f{"delete-edges-preserves-other-edges"};
        const auto id = MixedDeleteFixture(f);
        const auto [entity, instance] = f.AddEditable(id, Element::Edge);
        const auto original = f.ActiveMesh();
        const auto face = *original.faces().begin();
        const auto face_positions = FacePositions(original, face);
        std::vector<uint32_t> selected;
        for (const auto e : original.edges()) {
            const auto h = original.GetHalfedge(e, 0u);
            if (MeshEdgeUsers::Key(*original.GetFromVertex(h) - original.VertexFirst(), *original.GetToVertex(h) - original.VertexFirst()) == MeshEdgeUsers::Key(5u, 6u))
                selected.push_back(*e - original.EdgeFirst());
        }
        f.SelectElements(entity, selected, Element::Edge);
        f.Do(action::mesh::Delete{keep_vertices ? action::mesh::DeleteMode::OnlyEdgesAndFaces : action::mesh::DeleteMode::Edges});
        f.Checkpoint();
        ExpectCounts(f.ActiveMesh(), {keep_vertices ? 8u : 6u, 6u, 1u, 2u});
        expect(FacePositions(f.ActiveMesh(), face) == face_positions);
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    }
}

// Duplicate copies the selection; split copies only its shared boundary.
void TestDuplicateGeometry() {
    for (const bool split : {false, true}) {
        Fixture f{split ? "split-geometry" : "duplicate-geometry"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {-1, 0, 0}, {4, 0, 0}};
        data.AddFace(std::array{0u, 1u, 2u});
        data.AddFace(std::array{0u, 2u, 3u});
        const std::array<std::array<uint32_t, 2>, 1> wires{{{0u, 4u}}};
        const auto id = CreateMixedFixture(f.R, data, wires, true);
        const auto [entity, instance] = f.AddEditable(id, Element::Face);
        f.SelectElements(entity, std::array{0u}, Element::Face);
        const auto original = f.ActiveMesh();
        const auto untouched = original.FaceAt(1u);
        const auto untouched_positions = FacePositions(original, untouched);
        if (split) f.Stage(action::mesh::Split{});
        else f.Stage(action::mesh::Duplicate{});
        const vec3 shift{0, 0, 1};
        f.Stage(action::view::TransformElements{{.P = shift}});
        f.Commit();
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, split ? Counts{8u, 7u, 2u, 2u} : Counts{9u, 9u, 3u, 3u});
        expect(FacePositions(mesh, untouched) == untouched_positions);
        expect(f.Selection(Element::Vertex).Count() == 3u);
        expect(f.Selection(Element::Face).Count() == 1u);
        f.Selection(Element::Vertex).ForEach([&](uint32_t v) { expect(std::abs(mesh.GetPosition(he::VH{v}).z - 1.f) < 1e-5f); });
        for (const auto face : mesh.faces())
            for (const auto h : mesh.fh_range(face)) {
                const auto &a = f.R.Context.get<const MeshStore>().Arenas();
                expect(a.CornerColors.Get(*h) == vec4{.2f, .4f, .6f, 1.f});
                expect(a.CornerUvs[0].Get(*h) == vec2{.3f, .7f});
            }
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    }
}

void TestDeleteVerticesAndFaces() {
    for (const bool vertices : {true, false}) {
        Fixture f{"delete-vertices-and-faces"};
        const auto id = MixedDeleteFixture(f);
        const auto element = vertices ? Element::Vertex : Element::Face;
        const auto [entity, instance] = f.AddEditable(id, element);
        f.SelectElements(entity, std::array{0u}, element);
        f.Do(action::mesh::Delete{vertices ? action::mesh::DeleteMode::Vertices : action::mesh::DeleteMode::Faces});
        f.Checkpoint();
        ExpectCounts(f.ActiveMesh(), {7u, vertices ? 4u : 3u, 0u, 0u});
        expect(f.Selection(element).Count() == 0u);
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    }
}

void TestDeleteLoose() {
    Fixture f{"delete-loose"};
    const auto id = MixedDeleteFixture(f);
    const auto [entity, instance] = f.AddEditable(id, Element::Edge);
    const auto original = f.ActiveMesh();
    const auto face = *original.faces().begin();
    const auto face_positions = FacePositions(original, face);
    he::EH hidden;
    for (const auto e : original.edges()) {
        const auto h = original.GetHalfedge(e, 0u);
        if (!original.GetConnectivity().FaceOf(h) && MeshEdgeUsers::Key(*original.GetFromVertex(h) - original.VertexFirst(), *original.GetToVertex(h) - original.VertexFirst()) == MeshEdgeUsers::Key(0u, 1u)) hidden = e;
    }
    expect(bool(hidden));
    f.SelectElements(entity, std::array{*hidden - original.EdgeFirst()}, Element::Edge);
    f.Do(action::mesh::Hide{});
    f.Do(action::view::SetEditMode{.Mode = Element::Vertex});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::Loose});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {4u, 5u, 1u, 2u});
    expect(FacePositions(mesh, face) == face_positions);
    const auto &meshes = f.R.Context.get<const MeshStore>();
    expect(meshes.GetHiddenElements(id, Element::Edge).Contains(*hidden));
    expect(!mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(hidden, 0u)));
    ExpectRenderedGeometry(meshes, mesh);
}

void TestWireHullPreservesConnectivity() {
    Fixture f{"wire-hull-preserves-connectivity"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {5, 0, 0}, {6, 0, 0}};
    source.Data.Edges = {{0u, 1u}, {1u, 2u}, {4u, 5u}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    const std::array selected{0u, 1u, 2u, 3u};
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select hull vertices", {});
    f.Checkpoint();
    const auto &meshes = f.R.Context.get<const MeshStore>();
    const auto &render = meshes.Render();
    const auto original = Positions(f.ActiveMesh());
    struct Wire {
        uint32_t Edge, Halfedge, Opposite, Meshlet;
    };
    std::vector<Wire> wires;
    for (const auto e : f.ActiveMesh().edges()) {
        const auto h = f.ActiveMesh().GetHalfedge(e, 0u);
        wires.push_back({*e, *h, *f.ActiveMesh().GetOppositeHalfedge(h), render.ElementMeshlets[1].Get(*e)});
    }
    const auto check = [&](const state::Scene &, bool faces) {
        const auto mesh = f.ActiveMesh();
        const auto &owner = meshes.Get(id);
        ExpectCounts(mesh, {6u, faces ? 9u : 3u, faces ? 4u : 0u, faces ? 4u : 0u});
        ExpectPositions(mesh, original);
        expect(owner.RenderTopologies == (faces ? 3u : 6u));
        for (const auto &wire : wires) {
            expect(mesh.GetHalfedge(he::EH{wire.Edge}, 0u) == he::HH{wire.Halfedge});
            expect(mesh.GetOppositeHalfedge(he::HH{wire.Halfedge}) == he::HH{wire.Opposite});
            expect(!mesh.GetConnectivity().FaceOf(he::HH{wire.Halfedge}));
            expect(render.ElementMeshlets[1].Get(wire.Edge) == wire.Meshlet);
            expect(render.ActiveMeshlets.Contains(owner.MeshletRoot, wire.Meshlet));
        }
    };
    check(f.R, false);
    f.Do(action::mesh::ConvexHull{});
    f.Checkpoint();
    check(f.R, true);
}

void TestExtrudeEdgesMixed() {
    for (const bool surface : {false, true}) {
        Fixture f{"extrude-edges-mixed"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 1, 0}, {1, -1, 0}, {0, -2, 0}, {4, 0, 0}, {5, 0, 0}, {6, 1, 0}};
        if (surface) {
            data.AddFace(std::array{0u, 1u, 2u});
            data.AddFace(std::array{1u, 0u, 4u});
            data.AddFace(std::array{0u, 1u, 5u});
        }
        const std::vector<std::array<uint32_t, 2>> wires = surface ?
            std::vector<std::array<uint32_t, 2>>{{1u, 3u}, {3u, 4u}, {6u, 7u}, {0u, 1u}} :
            std::vector<std::array<uint32_t, 2>>{{0u, 1u}, {1u, 2u}, {1u, 3u}, {6u, 7u}};
        const auto id = CreateMixedFixture(f.R, data, wires, surface);
        const auto [entity, instance] = f.AddEditable(id, Element::Edge);
        const auto original = f.ActiveMesh();
        const auto before = CountsOf(original);
        std::vector<uint32_t> selected;
        for (const auto e : original.edges()) {
            const auto h = original.GetHalfedge(e, 0u);
            const auto key = MeshEdgeUsers::Key(*original.GetFromVertex(h) - original.VertexFirst(), *original.GetToVertex(h) - original.VertexFirst());
            const bool wire = !original.GetConnectivity().FaceOf(h);
            if (surface ? (wire ? key == MeshEdgeUsers::Key(1u, 3u) : key == MeshEdgeUsers::Key(0u, 1u)) :
                          key == MeshEdgeUsers::Key(0u, 1u) || key == MeshEdgeUsers::Key(1u, 2u)) selected.push_back(*e - original.EdgeFirst());
        }
        std::map<he::FH, std::vector<vec3>> faces;
        for (const auto face : original.faces()) faces.emplace(face, FacePositions(original, face));
        f.SelectElements(entity, selected, Element::Edge);
        f.Stage(action::mesh::Extrude{action::mesh::ExtrudeMode::Edges});
        f.Stage(action::view::TransformElements{{.P = vec3{0, 0, 1}}});
        f.Commit();
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {before.Vertices + 3u, before.Edges + 5u, before.Faces + 2u, before.Triangles + 4u});
        expect(f.Selection(Element::Vertex).Count() == 3u);
        expect(f.Selection(Element::Edge).Count() == 2u);
        for (const auto &[face, positions] : faces) expect(FacePositions(mesh, face) == positions);
        for (const auto face : mesh.faces())
            if (!faces.contains(face)) {
                expect(mesh.GetValence(face) == 4u);
                uint32_t moved = 0u;
                for (const auto v : mesh.fv_range(face)) moved += f.Selection(Element::Vertex).Contains(*v);
                expect(moved == 2u);
            }
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    }
}

void TestExtrudeRegionMixed() {
    Fixture f{"extrude-region-mixed"};
    MeshData data;
    data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {3, 0, 0}, {4, 0, 0}, {5, 1, 0}};
    data.AddFace(std::array{0u, 1u, 2u, 3u});
    const std::array<std::array<uint32_t, 2>, 1> wires{{{4u, 5u}}};
    const auto id = CreateMixedFixture(f.R, data, wires, true);
    f.AddEditable(id, Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::ExtrudeRepeat{3u, vec3{0, 0, 2}});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {28u, 41u, 17u, 34u});
    expect(f.Selection(Element::Vertex).Count() == 7u);
    expect(f.Selection(Element::Face).Count() == 1u);
    f.Selection(Element::Vertex).ForEach([&](uint32_t v) { expect(std::abs(mesh.GetPosition(he::VH{v}).z - 6.f) < 1e-5f); });
    // The isolated point creates a wire chain, while the face and wire create walls.
    uint32_t loose = 0u;
    for (const auto e : mesh.edges()) loose += !mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(e, 0u));
    expect(loose == 3u);
    ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
}

void TestExtrudeRegionSpinAttributes() {
    for (const bool closed : {false, true}) {
        Fixture f{"extrude-region-spin-attributes"};
        MeshSource source;
        for (uint32_t y = 0u; y < 3u; ++y)
            for (uint32_t x = 0u; x < 3u; ++x) source.Data.Positions.push_back({float(x), float(y), 0.f});
        for (uint32_t y = 0u; y < 2u; ++y)
            for (uint32_t x = 0u; x < 2u; ++x) {
                const auto v = 3u * y + x;
                source.Data.AddFace(std::array{v, v + 1u, v + 4u, v + 3u});
            }
        if (closed) {
            source.Data = MeshData{};
            source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            source.Data.AddFace(std::array{0u, 2u, 1u});
            source.Data.AddFace(std::array{0u, 1u, 3u});
            source.Data.AddFace(std::array{1u, 2u, 3u});
            source.Data.AddFace(std::array{2u, 0u, 3u});
        }
        source.Attrs.TexCoords0.emplace();
        source.Attrs.Colors0.emplace();
        source.Attrs.Colors0ComponentCount = 4u;
        for (const auto p : source.Data.Positions) {
            source.Attrs.TexCoords0->push_back({p.x / 3.f, p.y / 3.f});
            source.Attrs.Colors0->push_back({p.x / 3.f, p.y / 3.f, .5f, 1.f});
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        auto &meshes = f.R.Context.get<MeshStore>();
        const auto &a = meshes.Arenas();
        f.AddEditable(id, Element::Face);
        f.Do(action::selection::SelectAll{});
        const auto initial = f.ActiveMesh();
        const auto normal = Normalize(vec3{.2f, .3f, 1.f});
        std::vector<CustomNormal> offsets(initial.HalfEdgeCount());
        const auto view = meshes.GetCornerNormalView(id);
        const auto first = a.FaceCorners.First(meshes.Get(id).FaceCorners);
        for (const auto face : initial.faces())
            for (const auto h : initial.fh_range(face))
                offsets[*h - first].Offset = EncodeNormalOffset(normal, ComputeCornerFrame(view[*h], initial, *h));
        meshes.SetCustomCornerNormals(id, offsets);
        f.P->History.Commit("Author region normals", {});
        f.Checkpoint();
        const vec3 center{.5f, .5f, 0.f};
        f.Do(action::mesh::Spin{.Steps = 3u, .Angle = std::numbers::pi_v<float> * 1.5f, .Center = center, .Offset = 6.f});
        f.Checkpoint();
        const auto check = [&]() {
            const auto mesh = f.ActiveMesh();
            const auto normals = meshes.GetCornerNormalView(id);
            // Four bottom and four top quads, plus eight boundary quads per step.
            ExpectCounts(mesh, closed ? Counts{16u, 24u, 16u, 16u} : Counts{34u, 64u, 32u, 64u});
            expect(meshes.GetSelectedElements(id, Element::Vertex).Count() == (closed ? 4u : 9u));
            expect(meshes.GetSelectedElements(id, Element::Face).Count() == 4u);
            for (const auto face : mesh.faces()) {
                const auto positions = FacePositions(mesh, face);
                const bool reversed = closed ? positions.front().z < 5.5f : std::ranges::all_of(positions, [](vec3 p) { return std::abs(p.z) < 1e-5f; });
                for (const auto h : mesh.fh_range(face)) {
                    const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                    const auto layer = uint32_t(std::floor((p.z + 1e-4f) / 2.f));
                    expect(layer <= 3u);
                    auto original = p - center, direction = normal;
                    for (uint32_t i = 0u; i < layer; ++i) {
                        original = {original.y, -original.x, original.z};
                        direction = {-direction.y, direction.x, direction.z};
                    }
                    original += center;
                    const auto uv = a.CornerUvs[0].Get(*h);
                    const auto color = a.CornerColors.Get(*h);
                    expect(std::abs(uv.x - original.x / 3.f) < 1e-5f && std::abs(uv.y - original.y / 3.f) < 1e-5f);
                    expect(std::abs(color.x - original.x / 3.f) < 1e-5f && std::abs(color.y - original.y / 3.f) < 1e-5f);
                    expect(Length(normals[*h] - (reversed ? -direction : direction)) < 1e-4f) << "spin authored normal" << layer << reversed;
                }
            }
        };
        check();
    }
}

void TestExtrudeVertices() {
    Fixture f{"extrude-vertices"};
    MeshData data;
    data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 1, 0}, {3, 2, 0}, {4, 3, 0}};
    data.AddFace(std::array{0u, 1u, 2u});
    const std::array<std::array<uint32_t, 2>, 1> wires{{{2u, 3u}}};
    const auto id = CreateMixedFixture(f.R, data, wires, true);
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    f.SelectElements(entity, std::array{0u, 3u, 4u}, Element::Vertex);
    const auto original = Positions(f.ActiveMesh());
    const auto face = *f.ActiveMesh().faces().begin();
    const auto face_positions = FacePositions(f.ActiveMesh(), face);
    f.Stage(action::mesh::Extrude{action::mesh::ExtrudeMode::Vertices});
    const vec3 shift{0, 0, 1};
    f.Stage(action::view::TransformElements{{.P = shift}});
    f.Commit();
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {9u, 7u, 1u, 1u});
    expect(FacePositions(mesh, face) == face_positions);
    for (const auto &[v, p] : original) expect(Near(mesh.GetPosition(v), p));
    expect(f.Selection(Element::Vertex).Count() == 3u);
    expect(f.Selection(Element::Edge).Count() == 0u);
    f.Selection(Element::Vertex).ForEach([&](uint32_t v) {
        uint32_t incident = 0u;
        for (const auto e : mesh.edges()) {
            const auto h = mesh.GetHalfedge(e, 0u);
            const auto from = mesh.GetFromVertex(h), to = mesh.GetToVertex(h);
            if (*from != v && *to != v) continue;
            ++incident;
            expect(!mesh.GetConnectivity().FaceOf(h));
            expect(Near(mesh.GetPosition(he::VH{v}), mesh.GetPosition(*from == v ? to : from) + shift));
        }
        expect(incident == 1u);
    });
    ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
}

void TestCreateGeometry() {
    enum Kind { Edge,
                WireFace,
                CoincidentFace };
    for (const auto kind : {Edge, WireFace, CoincidentFace}) {
        Fixture f{"create-geometry"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {4, 0, 0}, {5, 0, 0}};
        source.Data.Edges = {{4u, 5u}};
        if (kind == WireFace) source.Data.Edges.insert(source.Data.Edges.end(), {{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 0u}});
        auto &meshes = f.R.Context.get<MeshStore>();
        uint32_t id;
        if (kind == CoincidentFace) {
            source.Data.AddFace(std::array{0u, 1u, 2u});
            source.Data.Edges.insert(source.Data.Edges.end(), {{2u, 3u}, {0u, 2u}, {0u, 3u}});
            id = CreateMixedFixture(f.R, source.Data, source.Data.Edges, false);
        } else id = CreateMesh(f.R, std::move(source)).StoreId;
        std::ranges::fill(meshes.EditEdgeSharpness(id), 1u);
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        const auto selected = kind == Edge ? std::vector<uint32_t>{0u, 3u} : kind == CoincidentFace ? std::vector<uint32_t>{0u, 2u, 3u} :
                                                                                                      std::vector<uint32_t>{0u, 1u, 2u, 3u};
        ApplyEditSelectionLists(f.R, std::array{std::pair{entity, std::span<const uint32_t>{selected}}}, Element::Vertex);
        if (kind == CoincidentFace) {
            f.Do(action::view::SetEditMode{.Mode = Element::Edge});
            const auto mesh = f.ActiveMesh();
            std::vector<uint32_t> wire_edges;
            for (const auto e : mesh.edges()) {
                const auto h = mesh.GetHalfedge(e, 0u);
                if (!mesh.GetConnectivity().FaceOf(h) && *mesh.GetFromVertex(h) < mesh.VertexFirst() + 4u) wire_edges.push_back(*e - mesh.EdgeFirst());
            }
            ApplyEditSelectionLists(f.R, std::array{std::pair{entity, std::span<const uint32_t>{wire_edges}}}, Element::Edge);
        }
        f.P->Settle();
        f.P->History.Commit("Select creation vertices", {});
        f.Checkpoint();
        const auto positions = Positions(f.ActiveMesh());
        struct OriginalEdge {
            he::EH Handle;
            uint64_t Ends;
            bool Face;
        };
        std::vector<OriginalEdge> edges;
        for (const auto e : f.ActiveMesh().edges()) {
            const auto mesh = f.ActiveMesh();
            const auto h = mesh.GetHalfedge(e, 0u);
            edges.push_back({e, MeshEdgeUsers::Key(*mesh.GetFromVertex(h), *mesh.GetToVertex(h)), bool(mesh.GetConnectivity().FaceOf(h))});
        }
        const auto check = [&] {
            const auto mesh = f.ActiveMesh();
            const auto expected = kind == Edge ? Counts{6u, 2u, 0u, 0u} : kind == CoincidentFace ? Counts{6u, 7u, 2u, 2u} :
                                                                                                   Counts{6u, 5u, 1u, 2u};
            ExpectCounts(mesh, expected);
            ExpectPositions(mesh, positions);
            for (const auto &edge : edges) {
                const auto h = mesh.GetHalfedge(edge.Handle, 0u);
                expect(MeshEdgeUsers::Key(*mesh.GetFromVertex(h), *mesh.GetToVertex(h)) == edge.Ends);
                expect(meshes.Arenas().EdgeSharpness.Get({*edge.Handle, 1u})[0] == 1u);
                const bool unrelated = uint32_t(edge.Ends >> 32u) == *positions[4].first;
                const bool coincident = kind == CoincidentFace && !edge.Face && edge.Ends == MeshEdgeUsers::Key(*positions[0].first, *positions[2].first);
                expect(bool(mesh.GetConnectivity().FaceOf(h)) == (edge.Face || (kind != Edge && !unrelated && !coincident)));
                if (unrelated) expect(bool(mesh.GetOppositeHalfedge(h)));
            }
            expect(f.Selection(Element::Vertex).Count() == selected.size());
            expect(f.Selection(Element::Face).Count() == (kind == Edge ? 0u : 1u));
            expect(f.Selection(Element::Edge).Count() == (kind == Edge ? 1u : selected.size() + (kind == CoincidentFace))) << "creation kind" << kind << "selected edges" << f.Selection(Element::Edge).Count();
            for (const auto face : mesh.faces()) {
                const auto points = FacePositions(mesh, face);
                vec3 area{};
                for (uint32_t i = 0u; i < points.size(); ++i) area += Cross(points[i], points[(i + 1u) % points.size()]);
                expect(std::abs(area.z) > 0.9f);
            }
        };

        f.Do(action::mesh::Fill{});
        f.Checkpoint();
        check();

        // Repeating F on the resulting edge or single face must not duplicate it.
        f.Do(action::mesh::Fill{});
        f.Checkpoint();
        check();
    }
}

void TestLimitedDissolveChains() {
    for (const bool branch : {false, true}) {
        Fixture f{"limited-dissolve-chains"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {2, 2, 0}, {10, 0, 0}, {11, 0, 0}, {10, 1, 0}};
        data.AddFace(std::array{5u, 6u, 7u});
        std::vector<std::array<uint32_t, 2>> wires{{0u, 1u}, {1u, 2u}, {2u, 3u}, {3u, 4u}};
        if (branch) wires.push_back({1u, 3u});
        const auto id = CreateMixedFixture(f.R, data, wires, false);
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        f.SelectElements(entity, std::array{0u, 1u, 2u, 3u, 4u}, Element::Vertex);
        const auto face = *f.ActiveMesh().faces().begin();
        const auto positions = FacePositions(f.ActiveMesh(), face);
        f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .Angle = .0872665f});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, branch ? Counts{8u, 8u, 1u, 1u} : Counts{6u, 5u, 1u, 1u});
        expect(FacePositions(mesh, face) == positions);
        if (!branch) {
            std::set<uint32_t> remaining;
            for (const auto v : mesh.vertices())
                if (*v - mesh.VertexFirst() < 5u) remaining.insert(*v - mesh.VertexFirst());
            expect(remaining == std::set<uint32_t>{0u, 2u, 4u});
        }
    }
}

void TestDissolveMixed() {
    const std::array<std::array<std::array<uint32_t, 2>, 3>, 3> wires{{
        {{{0u, 1u}, {1u, 5u}, {5u, 6u}}}, // A chain joins its kept endpoints.
        {{{0u, 1u}, {1u, 5u}, {1u, 6u}}}, // A branch cannot be dissolved.
        {{{0u, 1u}, {1u, 5u}, {5u, 0u}}}, // A join reuses its existing closing wire.
    }};
    for (uint32_t kind = 0u; kind < wires.size(); ++kind) {
        Fixture f{"dissolve-mixed"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}, {1, -1, 0}, {1, -2, 0}, {5, 0, 0}, {6, 0, 0}, {5, 1, 0}, {9, 9, 0}};
        data.AddFace(std::array{0u, 2u, 3u, 4u});
        data.AddFace(std::array{7u, 8u, 9u});
        const auto id = CreateMixedFixture(f.R, data, wires[kind], false);
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        const auto original = f.ActiveMesh();
        const auto first = original.VertexFirst();
        std::vector<std::pair<he::FH, std::vector<vec3>>> faces;
        for (const auto face : original.faces()) faces.emplace_back(face, FacePositions(original, face));
        he::EH closing;
        for (const auto edge : original.edges()) {
            const auto h = original.GetHalfedge(edge, 0u);
            if (MeshEdgeUsers::Key(*original.GetFromVertex(h) - first, *original.GetToVertex(h) - first) == MeshEdgeUsers::Key(0u, 5u)) closing = edge;
        }
        f.SelectElements(entity, kind == 0u ? std::vector<uint32_t>{1u, 5u, 10u} : std::vector<uint32_t>{1u, 10u}, Element::Vertex);
        f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Vertices});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {kind == 0u ? 8u : kind == 1u ? 10u :
                                                           9u,
                            kind == 1u ? 10u : 8u, 2u, 3u});
        for (const auto &[face, positions] : faces) expect(FacePositions(mesh, face) == positions);
        for (const auto vertex : mesh.vertices()) {
            const auto index = *vertex - first;
            expect(index != 10u && (kind == 1u || index != 1u) && (kind != 0u || index != 5u));
            expect(Near(mesh.GetPosition(vertex), data.Positions[index]));
        }
        std::set<uint64_t> actual;
        for (const auto edge : mesh.edges()) {
            const auto h = mesh.GetHalfedge(edge, 0u);
            if (mesh.GetConnectivity().FaceOf(h)) continue;
            actual.insert(MeshEdgeUsers::Key(*mesh.GetFromVertex(h) - first, *mesh.GetToVertex(h) - first));
            if (kind == 2u) expect(edge == closing);
        }
        const std::set<uint64_t> expected = kind == 0u ? std::set{MeshEdgeUsers::Key(0u, 6u)} : kind == 1u ? std::set{MeshEdgeUsers::Key(0u, 1u), MeshEdgeUsers::Key(1u, 5u), MeshEdgeUsers::Key(1u, 6u)} :
                                                                                                             std::set{MeshEdgeUsers::Key(0u, 5u)};
        expect(actual == expected);
    }
}

void TestMergeMixed() {
    using Mode = action::mesh::MergeMode;
    for (const auto mode : {Mode::Center, Mode::First, Mode::Last, Mode::Collapse, Mode::ByDistance}) {
        Fixture f{"merge-mixed"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {.01f, 0, 0}, {2, 1, 0}, {5, 0, 0}, {6, 0, 0}, {5, 1, 0}};
        data.AddFace(std::array{0u, 1u, 2u});
        data.AddFace(std::array{3u, 4u, 5u});
        const std::array<std::array<uint32_t, 2>, 2> wires{{{0u, 2u}, {1u, 2u}}};
        const auto id = CreateMixedFixture(f.R, data, wires, true);
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        f.SelectElements(entity, std::array{0u, 1u}, Element::Vertex);
        const auto untouched = f.ActiveMesh().FaceAt(1u);
        const auto positions = FacePositions(f.ActiveMesh(), untouched);
        f.Do(action::mesh::Merge{.Mode = mode, .Distance = .02f});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {5u, 4u, 1u, 1u});
        expect(FacePositions(mesh, untouched) == positions);
        expect(f.Selection(Element::Vertex).Count() == 1u);
        const auto target = mode == Mode::Last ? data.Positions[1] : mode == Mode::Center || mode == Mode::Collapse ? (data.Positions[0] + data.Positions[1]) * .5f :
                                                                                                                      data.Positions[0];
        f.Selection(Element::Vertex).ForEach([&](uint32_t v) { expect(Near(mesh.GetPosition(he::VH{v}), target)); });
        uint32_t loose = 0u;
        for (const auto e : mesh.edges()) loose += !mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(e, 0u));
        expect(loose == 1u); // The collapsed face and coincident wires share one surviving edge.
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    }
}

void TestSubdivideMixed() {
    for (const bool surface : {false, true}) {
        Fixture f{"subdivide-mixed"};
        MeshData data;
        data.Positions = {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}, {-2, 0, 0}};
        if (surface) {
            data.AddFace(std::array{0u, 1u, 2u});
            data.AddFace(std::array{1u, 0u, 3u});
            data.AddFace(std::array{0u, 1u, 4u});
        }
        const std::array<std::array<uint32_t, 2>, 2> wires{{{0u, 1u}, {0u, 1u}}};
        const auto id = CreateMixedFixture(f.R, data, wires, true);
        const auto [entity, instance] = f.AddEditable(id, Element::Edge);
        const auto original = f.ActiveMesh();
        const auto before = CountsOf(original);
        std::vector<uint32_t> selected;
        bool wire_selected = false;
        for (const auto e : original.edges()) {
            const auto h = original.GetHalfedge(e, 0u);
            if (MeshEdgeUsers::Key(*original.GetFromVertex(h) - original.VertexFirst(), *original.GetToVertex(h) - original.VertexFirst()) != MeshEdgeUsers::Key(0u, 1u)) continue;
            const bool wire = !original.GetConnectivity().FaceOf(h);
            if (wire && wire_selected) continue;
            wire_selected |= wire;
            selected.push_back(*e - original.EdgeFirst());
        }
        f.SelectElements(entity, selected, Element::Edge);
        f.Do(action::mesh::Subdivide{3u});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        const auto added = 3u * uint32_t(selected.size());
        ExpectCounts(mesh, {before.Vertices + added, before.Edges + added, before.Faces, before.Triangles + (surface ? 9u : 0u)});
        expect(f.Selection(Element::Edge).Count() == 4u * selected.size());
        std::array<uint32_t, 3> intermediate{};
        for (const auto v : mesh.vertices()) {
            const auto position = mesh.GetPosition(v);
            for (uint32_t i = 0u; i < 3u; ++i) intermediate[i] += Near(position, vec3{.5f * float(i + 1u), 0, 0});
        }
        for (const auto count : intermediate) expect(count == selected.size());
        for (const auto face : mesh.faces()) {
            expect(mesh.GetValence(face) == 6u);
            for (const auto h : mesh.fh_range(face)) expect(f.R.Context.get<const MeshStore>().Arenas().CornerUvs[0].Get(*h) == vec2{.3f, .7f});
        }
        ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    }
}

void TestLineSubdivide() {
    Fixture f{"line-subdivide"};
    f.Checkpoint();
    const auto axis = Normalize(Cross(Normalize(f.FrameView().CameraPosition), vec3{0.f, 0.f, 1.f}));
    MeshSource source;
    source.Data.Positions = {-3.f * axis, -1.f * axis, axis, 3.f * axis};
    source.Data.Edges = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto instance = AddMesh(f.R, id, MeshInstanceCreateInfo{}).second;
    f.P->Settle();
    f.Do(action::selection::Select{instance});
    f.EnterEdit(Element::Edge);
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 3u, 0u, 0u});
    ExpectRenderTopology(f, id, 1u, 3u);

    f.Click(f.ViewFraction(vec3{0.f}));
    expect(f.Selection(Element::Edge).Count() == 1u);
    f.Do(action::mesh::Subdivide{1u});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {5u, 4u, 0u, 0u});
    expect(f.Selection(Element::Edge).Count() == 2u);

    const std::array ExpectedCoordinates{-3.f, -1.f, 0.f, 1.f, 3.f};
    std::vector<float> coordinates;
    for (const auto vertex : mesh.vertices()) {
        const auto position = mesh.GetPosition(vertex);
        const auto coordinate = Dot(position, axis);
        coordinates.push_back(coordinate);
        expect(Near(position, coordinate * axis));
    }
    std::ranges::sort(coordinates);
    expect(coordinates.size() == ExpectedCoordinates.size());
    for (uint32_t i = 0u; i < std::min(coordinates.size(), ExpectedCoordinates.size()); ++i)
        expect(std::abs(coordinates[i] - ExpectedCoordinates[i]) < 1e-5f);

    std::vector<std::pair<int32_t, int32_t>> segments;
    for (const auto edge : mesh.edges()) {
        const auto halfedge = mesh.GetHalfedge(edge, 0u);
        const auto a = int32_t(std::lround(Dot(mesh.GetPosition(mesh.GetFromVertex(halfedge)), axis)));
        const auto b = int32_t(std::lround(Dot(mesh.GetPosition(mesh.GetToVertex(halfedge)), axis)));
        segments.emplace_back(std::min(a, b), std::max(a, b));
    }
    std::ranges::sort(segments);
    const std::vector<std::pair<int32_t, int32_t>> ExpectedSegments{{-3, -1}, {-1, 0}, {0, 1}, {1, 3}};
    expect(segments == ExpectedSegments);
    ExpectRenderTopology(f, id, 1u, 4u);
}

// Independent edge-based reference: every edge contributes once at each endpoint,
// and each iteration reads the previous iteration's complete positions.

void TestSmoothVertices() {
    Fixture f{"smooth-vertices"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 2, 0}, {4, 0, 0}, {9, 9, 9}};
    source.Data.Edges = {{0u, 1u}, {1u, 2u}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    f.SelectElements(entity, std::array{1u, 3u}, Element::Vertex);
    const auto before = Positions(f.ActiveMesh());
    auto expected = before;
    expected[1].second.y = .5f; // Two half steps toward the endpoints' mean; the isolated point stays fixed.
    f.Do(action::mesh::SmoothVertices{.Factor = .5f, .Repeat = 2u, .X = false});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected);
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::SmoothVertices{.Factor = 0.f});
    expect(*f.P->History.Present == history);
}

void TestSparseSelectionLifecycle() {
    Fixture f{"selection-lifecycle"};
    MeshSource source;
    for (uint32_t i = 0u; i < 258u; ++i) source.Data.Positions.push_back({float(i), 0, 0});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    f.SelectElements(entity, std::array{256u, 257u}, Element::Vertex);
    f.P->History.Commit("Select last block", {});
    const auto check = [&](const state::Scene &scene, bool deleted) {
        const auto &meshes = scene.Context.get<const MeshStore>();
        expect(GetMesh(scene, entity).VertexCount() == (deleted ? 256u : 258u));
        expect(meshes.GetSelectedElements(id, Element::Vertex).Count() == (deleted ? 0u : 2u));
        ExpectSelectionIndex(meshes, id);
    };
    check(f.R, false);
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::Vertices});
    f.Checkpoint();
    check(f.R, true);
    f.CheckUndoRedo(check);
    f.CheckSaved(check);
}

void TestMakePlanarFaces() {
    Fixture f{"planar-faces"};
    MeshSource source;
    source.Data.Positions = {{-1, -1, 1}, {1, -1, -1}, {1, 1, 1}, {-1, 1, -1}, {3, 0, 2}, {4, 0, 2}, {3, 1, 2}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u});
    source.Data.AddFace(std::array{4u, 5u, 6u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Face);
    f.Do(action::selection::SelectAll{});
    auto expected = Positions(f.ActiveMesh());
    for (uint32_t i = 0u; i < 4u; ++i) expected[i].second.z = 0.f;
    f.Do(action::mesh::MakePlanarFaces{});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected); // The saddle projects onto z=0; the triangle is unchanged.
    ExpectCounts(f.ActiveMesh(), {7u, 7u, 2u, 3u});
}

// A closed, consistently wound wire has two oppositely directed users per edge.
void ExpectClosedWire(const Mesh &mesh) {
    double volume = 0;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u), opposite = mesh.GetOppositeHalfedge(h);
        expect(bool(opposite));
        if (opposite) {
            expect(mesh.GetToVertex(h) == mesh.GetFromVertex(opposite));
            expect(mesh.GetFromVertex(h) == mesh.GetToVertex(opposite));
        }
    }
    for (const auto face : mesh.faces()) {
        const auto points = FacePositions(mesh, face);
        expect(points.size() == 4u);
        for (size_t i = 1; i + 1 < points.size(); ++i) volume += Dot(points[0], Cross(points[i], points[i + 1])) / 6.;
        expect(std::isfinite(Length(mesh.GetNormal(face))));
    }
    expect(volume > 0.0) << "wire volume" << volume;
}

void TestWireframeQuad() {
    Fixture f{"wireframe-quad"};
    MeshSource source;
    source.Data.Positions = {{-1.f, -1.f, 0.f}, {1.f, -1.f, 0.f}, {1.f, 1.f, 0.f}, {-1.f, 1.f, 0.f}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u});
    source.Attrs.TexCoords0 = std::vector<vec2>{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}, {0.f, 1.f}};
    source.Attrs.Colors0 = std::vector<vec4>{{0.f, 0.f, 1.f, 1.f}, {1.f, 0.f, 1.f, 1.f}, {1.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 1.f, 1.f}};
    source.Attrs.Colors0ComponentCount = 4u;
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    const auto check = [&](float thickness, float offset, bool even, bool boundary, bool replace) {
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {uint32_t(12 + 4 * boundary + 4 * !replace), uint32_t(20 + 12 * boundary + 4 * !replace), uint32_t(8 + 8 * boundary + !replace), uint32_t(16 + 16 * boundary + 2 * !replace)});
        expect(f.Selection(Element::Face).Count() == uint32_t(8 + 8 * boundary));
        std::vector<vec3> expected;
        const float radius = thickness * 0.5f, mid = radius * offset;
        const float inset = even ? radius : radius / std::sqrt(2.f);
        for (const auto &[v, p] : original) {
            expected.push_back(p + vec3{0.f, 0.f, mid - radius});
            expected.push_back(p + vec3{0.f, 0.f, mid + radius});
            expected.push_back({p.x * (1.f - inset), p.y * (1.f - inset), mid});
            if (boundary) expected.push_back({p.x * (1.f + inset), p.y * (1.f + inset), mid});
            if (!replace) expected.push_back(p);
        }
        for (const auto v : mesh.vertices()) {
            const auto p = mesh.GetPosition(v);
            const auto found = std::ranges::find_if(expected, [&](vec3 q) { return Near(p, q); });
            expect(found != expected.end()) << "unexpected wire point" << p.x << p.y << p.z;
            if (found != expected.end()) expected.erase(found);
        }
        expect(expected.empty());
        const auto &arenas = f.R.Context.get<const MeshStore>().Arenas();
        for (const auto face : mesh.faces())
            for (const auto h : mesh.fh_range(face)) {
                const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                const vec2 uv{float(p.x > 0.f), float(p.y > 0.f)};
                expect(arenas.CornerUvs[0].Get(*h) == uv);
                const auto color = arenas.CornerColors.Get(*h);
                expect(color == vec4{uv.x, uv.y, 1.f, 1.f});
            }
        if (boundary && replace) ExpectClosedWire(mesh);
    };
    f.Do(action::mesh::Wireframe{.Thickness = 0.2f, .Offset = 0.5f});
    f.Checkpoint();
    check(0.2f, 0.5f, true, true, true);
    f.P->Undo();
    f.Do(action::mesh::Wireframe{.Thickness = 0.2f, .Offset = -1.f, .Replace = false, .Boundary = false, .Even = false});
    f.Checkpoint();
    check(0.2f, -1.f, false, false, false);
    f.P->Undo();
    f.Do(action::mesh::Wireframe{.Thickness = 0.1f, .Even = false, .Relative = true});
    f.Checkpoint();
    check(0.2f, 0.f, false, true, true); // Every source edge is two units long.
}

void TestWireframeRegion() {
    Fixture f{"wireframe-region"};
    f.Cube(Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    f.Do(action::mesh::Wireframe{.Thickness = 0.2f});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {40u, 96u, 48u, 96u});
    ExpectClosedWire(f.ActiveMesh());
    f.P->Undo();
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), original);
    const auto entity = GetActiveMeshEntity(f.R);
    const std::array selected{0u};
    f.SelectElements(entity, selected, Element::Face);
    const auto mesh = f.ActiveMesh();
    std::vector<std::pair<he::FH, std::vector<vec3>>> untouched;
    for (const auto face : mesh.faces())
        if (!f.Selection(Element::Face).Contains(*face)) untouched.emplace_back(face, FacePositions(mesh, face));
    f.Do(action::mesh::Wireframe{.Thickness = 0.1f});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {24u, 44u, 21u, 42u});
    for (const auto &[face, points] : untouched) expect(FacePositions(f.ActiveMesh(), face) == points);
    for (const auto &[vertex, position] : original) expect(f.ActiveMesh().GetPosition(vertex) == position);
}

void TestSplitNonplanarFaces() {
    Fixture f{"split-nonplanar"};
    MeshSource source;
    // Two planar quads meeting at a ninety-degree fold, represented by one hexagon.
    source.Data.Positions = {{0.f, 0.f, 0.f}, {1.f, 0.f, 1.f}, {1.f, 2.f, 1.f}, {0.f, 2.f, 0.f}, {-1.f, 2.f, 1.f}, {-1.f, 0.f, 1.f}, {3.f, 0.f, 0.f}, {4.f, 0.f, 0.f}, {4.f, 1.f, 0.f}, {3.f, 1.f, 0.f}, {2.f, 0.f, 0.f}, {2.f, 2.f, 2.f}, {3.f, 2.f, 1.f}};
    for (const auto &face : std::vector<std::vector<uint32_t>>{{0, 1, 2, 3, 4, 5}, {6, 7, 8, 9}, {2, 1, 10, 11}, {8, 12, 9}}) source.Data.AddFace(face);
    std::vector<vec2> uvs;
    for (const auto p : source.Data.Positions) uvs.push_back({p.x, p.y});
    source.Attrs.TexCoords0 = uvs;
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Face);
    const std::array selected{0u, 1u, 3u};
    f.SelectElements(entity, selected, Element::Face);
    f.P->History.Commit("Select folded polygon planar polygon and triangle", {});
    const auto original = Positions(f.ActiveMesh());
    const auto counts = CountsOf(f.ActiveMesh());
    std::vector<std::pair<he::FH, std::vector<vec3>>> unchanged;
    for (const auto face : f.ActiveMesh().faces())
        if (f.ActiveMesh().GetValence(face) != 6u) unchanged.emplace_back(face, FacePositions(f.ActiveMesh(), face));
    const auto base = *f.P->History.Present;
    f.Do(action::mesh::SplitNonplanarFaces{.Angle = 2.f});
    f.Checkpoint();
    expect(*f.P->History.Present == base) << "split above threshold should not create history";
    ExpectCounts(f.ActiveMesh(), counts);
    const auto check = [&] {
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {counts.Vertices, counts.Edges + 1u, counts.Faces + 1u, counts.Triangles});
        ExpectPositions(mesh, original);
        for (const auto &[face, points] : unchanged) expect(FacePositions(mesh, face) == points);
        bool crease = false;
        for (const auto edge : mesh.edges()) {
            const auto h = mesh.GetHalfedge(edge, 0u);
            const auto a = mesh.GetPosition(mesh.GetFromVertex(h)), b = mesh.GetPosition(mesh.GetToVertex(h));
            if ((Near(a, original[0].second) && Near(b, original[3].second)) || (Near(a, original[3].second) && Near(b, original[0].second))) {
                crease = true;
                const auto opposite = mesh.GetOppositeHalfedge(h);
                expect(bool(opposite));
                if (opposite) expect(std::abs(Dot(mesh.GetNormal(mesh.GetFace(h)), mesh.GetNormal(mesh.GetFace(opposite)))) < 1e-5f);
            }
        }
        expect(crease);
        const auto &arenas = f.R.Context.get<const MeshStore>().Arenas();
        for (const auto face : mesh.faces())
            for (const auto h : mesh.fh_range(face)) {
                const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                expect(arenas.CornerUvs[0].Get(*h) == vec2{p.x, p.y});
            }
        expect(f.Selection(Element::Face).Count() == 4u);
    };
    f.Do(action::mesh::SplitNonplanarFaces{});
    f.Checkpoint();
    check();
    const auto split = *f.P->History.Present;
    f.Do(action::mesh::SplitNonplanarFaces{});
    f.Checkpoint();
    expect(*f.P->History.Present == split) << "planar output should not create history";
}

void TestSplitNonplanarConcave() {
    Fixture f{"split-nonplanar-concave"};
    MeshSource source;
    source.Data.Positions = {{0.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {2.f, 2.f, 0.f}, {1.f, 1.f, 1.f}, {0.f, 2.f, 0.f}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u, 4u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    const auto check = [&] {
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {5u, 7u, 3u, 3u});
        ExpectPositions(mesh, original);
        float area = 0.f;
        for (const auto face : mesh.faces()) {
            const auto p = FacePositions(mesh, face);
            expect(p.size() == 3u);
            const float signed_area = Cross(p[1] - p[0], p[2] - p[0]).z * 0.5f;
            expect(Length(Cross(p[1] - p[0], p[2] - p[0])) > 0.f);
            area += std::abs(signed_area);
        }
        expect(std::abs(area - 3.f) < 1e-5f);
        for (const auto edge : mesh.edges()) {
            const auto h = mesh.GetHalfedge(edge, 0u);
            const auto a = mesh.GetPosition(mesh.GetFromVertex(h)), b = mesh.GetPosition(mesh.GetToVertex(h));
            expect(!(Near(a, original[2].second) && Near(b, original[4].second)) && !(Near(b, original[2].second) && Near(a, original[4].second))) << "split crossed the concave opening";
        }
    };
    f.Do(action::mesh::SplitNonplanarFaces{});
    f.Checkpoint();
    check();
    f.P->Undo();
    f.Do(action::mesh::FlipNormals{});
    f.Do(action::mesh::SplitNonplanarFaces{});
    f.Checkpoint();
    check();
}
void TestSplitNonplanarLargePolygon() {
    Fixture f{"split-nonplanar-large"};
    MeshSource source;
    std::vector<uint32_t> loop;
    for (uint32_t i = 0u; i < 66u; ++i) {
        const float angle = -0.5f * std::numbers::pi_v<float> + float(i) * (2.f * std::numbers::pi_v<float> / 66.f);
        const float x = i == 0u || i == 33u ? 0.f : std::cos(angle);
        source.Data.Positions.push_back({x, std::sin(angle), std::abs(x)});
        loop.push_back(i);
    }
    source.Data.AddFace(loop);
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    f.Do(action::mesh::SplitNonplanarFaces{});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {66u, 67u, 2u, 64u});
    ExpectPositions(mesh, original);
    for (const auto face : mesh.faces()) {
        expect(mesh.GetValence(face) == 34u);
        const auto normal = mesh.GetNormal(face), center = mesh.CalcFaceCentroid(face);
        for (const auto vertex : mesh.fv_range(face)) expect(std::abs(Dot(mesh.GetPosition(vertex) - center, normal)) < 1e-5f);
    }
}
// Validate the render triangles independently of the triangulation algorithm:
// winding and area must match the polygon, and internal directed edges cancel.
template<typename Position>
void ExpectPolygonTessellation(const MeshStore &meshes, const Mesh &mesh, Position position) {
    const auto &arenas = meshes.Arenas();
    for (const auto face : mesh.faces()) {
        std::vector<vec3> points;
        for (const auto vertex : mesh.fv_range(face)) points.push_back(position(vertex));
        const auto origin = points.front();
        vec3 area{};
        for (uint32_t i = 1u; i + 1u < points.size(); ++i) area += Cross(points[i] - origin, points[i + 1u] - origin);
        const auto length = Length(area);
        expect(length > 0.f);
        if (!(length > 0.f)) continue;
        const auto normal = area / length;
        double triangle_area = 0.0;
        std::map<std::pair<uint32_t, uint32_t>, int> boundary;
        const auto edge = [&](uint32_t a, uint32_t b, int weight) { boundary[std::minmax(a, b)] += a < b ? weight : -weight; };
        std::set<uint32_t> corners;
        for (const auto h : mesh.fh_range(face)) {
            corners.insert(*h);
            edge(*mesh.GetConnectivity().Previous(h), *h, -1);
        }
        const auto first = arenas.FaceTriangles.Get({*face, 1u})[0];
        for (const auto triangle : arenas.Triangles.Get({first, uint32_t(points.size() - 2u)})) {
            for (uint32_t k = 0u; k < 3u; ++k) {
                expect(corners.contains(triangle[k]));
                edge(triangle[k], triangle[(k + 1u) % 3u], 1);
            }
            const auto a = position(mesh.GetToVertex(he::HH{triangle.x}));
            const auto b = position(mesh.GetToVertex(he::HH{triangle.y}));
            const auto c = position(mesh.GetToVertex(he::HH{triangle.z}));
            const auto signed_area = Dot(Cross(b - a, c - a), normal);
            expect(signed_area >= -length * 1e-5f) << "triangle winds outside polygon";
            triangle_area += std::abs(signed_area);
        }
        expect(std::abs(triangle_area - length) < length * 1e-4) << "render triangles overlap or leave a gap";
        for (const auto &[ends, count] : boundary) expect(count == 0) << "triangle edge differs from polygon boundary" << ends.first << ends.second;
    }
}

void ExpectPolygonTessellation(const MeshStore &meshes, const Mesh &mesh) {
    ExpectPolygonTessellation(meshes, mesh, [&](he::VH vertex) { return mesh.GetPosition(vertex); });
}

void TestPositionRetessellation() {
    Fixture f{"position-retessellation"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u});
    source.Data.Positions.insert(source.Data.Positions.end(), {{10, 0, 0}, {11, 0, 0}, {10, 1, 0}});
    source.Data.AddFace(std::array{4u, 5u, 6u});
    source.Attrs.TexCoords0.emplace();
    for (const auto p : source.Data.Positions) source.Attrs.TexCoords0->push_back({p.x, p.y});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    const std::array selected{1u};
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select polygon vertex", {});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    const auto original = Positions(mesh);
    const auto counts = CountsOf(mesh);
    const auto vertex = original[1].first;
    const auto target = vec3{.5f, 1.f, 0.f};
    const auto delta = target - original[1].second;
    const auto &meshes = f.R.Context.get<const MeshStore>();
    const auto &a = meshes.Arenas();
    const auto first = a.FaceTriangles.Get({*mesh.FaceAt(0u), 1u})[0];
    const auto base_triangles = a.Triangles.Get({first, 2u}) | std::ranges::to<std::vector>();
    const auto untouched = a.FaceTriangles.Get({*mesh.FaceAt(1u), 1u})[0];
    const auto untouched_triangle = a.Triangles.Get({untouched, 1u})[0];
    const auto check = [&](vec3 moved, bool preview) {
        const auto current = f.ActiveMesh();
        ExpectCounts(current, counts);
        auto expected = original;
        expected[1].second = moved;
        ExpectPositions(current, preview ? original : expected);
        const auto position = [&](he::VH v) { return v == vertex ? moved : current.GetPosition(v); };
        ExpectPolygonTessellation(meshes, current, position);
        expect(a.Triangles.Get({untouched, 1u})[0] == untouched_triangle);
        for (const auto face : current.faces())
            for (const auto h : current.fh_range(face)) {
                const auto v = current.GetToVertex(h);
                const auto uv = a.CornerUvs[0].Get(*h);
                const auto p = original[*v - current.VertexFirst()].second;
                expect(Length(uv - vec2{p.x, p.y}) < 1e-5f);
            }
        ExpectRenderedGeometry(meshes, current);
        if (preview) {
            const auto &scene = f.R.Context.get<const GpuSceneState>();
            const auto &buffers = f.R.Context.get<const GpuBuffers>();
            const auto posed = buffers.PosedPositions.View(scene.PosedByEntity.at(entity).PositionNamespace(0u));
            expect(Near(posed[*vertex], moved));
        }
    };
    check(original[1].second, false);
    f.Stage(action::view::TransformElements{{.P = delta}});
    f.Checkpoint();
    check(target, true);
    expect(!std::ranges::equal(a.Triangles.Get({first, 2u}), base_triangles));
    f.Cancel();
    f.Checkpoint();
    check(original[1].second, false);
    expect(std::ranges::equal(a.Triangles.Get({first, 2u}), base_triangles));
    f.Stage(action::view::TransformElements{{.P = delta}});
    f.Checkpoint();
    f.Commit();
    f.Checkpoint();
    check(target, false);
    f.P->Undo();
    f.Checkpoint();
    check(original[1].second, false);
    f.P->Redo();
    f.Checkpoint();
    check(target, false);
    f.P->Undo();
    const vec3 smoothed{.5f, 1.5f, 0.f};
    f.Do(action::mesh::SmoothVertices{.Factor = 1.5f});
    f.Checkpoint();
    check(smoothed, false);
    expect(f.P->Save());
    expect(f.P->Close());
    Engine reopened{false};
    expect(reopened.P->Open(f.Dir));
    reopened.P->Settle();
    ExpectPolygonTessellation(reopened.R.Context.get<const MeshStore>(), GetMesh(reopened.R, entity));
}

void TestDissolveCollapsedFaces() {
    Fixture f{"dissolve-collapsed-faces"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {4, 0, 0}, {5, 0, 0}, {4, 1, 0}};
    source.Data.AddFace(std::array{0u, 1u, 2u});
    source.Data.AddFace(std::array{3u, 4u, 5u});
    source.Attrs.TexCoords0 = std::vector<vec2>{{0, 0}, {1, 0}, {0, 1}, {4, 0}, {5, 0}, {4, 1}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    const auto first = f.ActiveMesh().VertexFirst();
    f.SelectElements(entity, std::array{0u}, Element::Vertex);
    f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Vertices});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {5u, 4u, 1u, 1u});
    uint32_t wires = 0u;
    for (const auto edge : mesh.edges()) {
        const auto h = mesh.GetHalfedge(edge, 0u);
        if (mesh.GetConnectivity().FaceOf(h)) continue;
        ++wires;
        expect(MeshEdgeUsers::Key(*mesh.GetFromVertex(h) - first, *mesh.GetToVertex(h) - first) == MeshEdgeUsers::Key(1u, 2u));
    }
    expect(wires == 1u);
    for (const auto face : mesh.faces())
        for (const auto h : mesh.fh_range(face)) {
            const auto position = mesh.GetPosition(mesh.GetToVertex(h));
            expect(f.R.Context.get<const MeshStore>().Arenas().CornerUvs[0].Get(*h) == vec2{position.x, position.y});
        }
    ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
}

void TestLimitedDissolveSelection() {
    for (const auto element : {Element::Vertex, Element::Edge, Element::Face}) {
        Fixture f{"limited-dissolve-selection"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u});
        source.Data.AddFace(std::array{0u, 2u, 3u});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, element);
        const auto mesh = f.ActiveMesh();
        std::vector<uint32_t> selected = element == Element::Vertex ? std::vector<uint32_t>{0u, 2u} : std::vector<uint32_t>{0u};
        if (element == Element::Edge)
            for (const auto edge : mesh.edges()) {
                const auto h = mesh.GetHalfedge(edge, 0u);
                if (mesh.GetOppositeHalfedge(h)) selected = {*edge - mesh.EdgeFirst()};
            }
        f.SelectElements(entity, selected, element);
        f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited});
        f.Checkpoint();
        ExpectCounts(f.ActiveMesh(), element == Element::Face ? Counts{4u, 5u, 2u, 2u} : Counts{4u, 4u, 1u, 2u});
    }
}

void TestDissolveDelimiters() {
    enum Delimiter { Material,
                     Sharp,
                     UV };
    for (const auto delimiter : {Material, Sharp, UV})
        for (const bool enabled : {false, true}) {
            Fixture f{"dissolve-delimiters"};
            MeshSource source;
            source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
            source.Data.AddFace(std::array{0u, 1u, 2u});
            source.Data.AddFace(std::array{0u, 2u, 3u});
            source.Primitives.MaterialIndices = {0u, 0u};
            source.Primitives.ElementPrimitiveIndices = {0u, delimiter == Material ? 1u : 0u};
            source.Attrs.TexCoords0 = std::vector<vec2>{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            const auto id = CreateMesh(f.R, std::move(source)).StoreId;
            const auto [entity, instance] = f.AddEditable(id, Element::Edge);
            const auto original = f.ActiveMesh();
            if (delimiter == UV) {
                auto &uv = f.R.Context.get<MeshStore>().Arenas().CornerUvs[0];
                for (const auto h : original.fh_range(original.FaceAt(1u))) uv.Values.Buffer.GetMutableSpan<vec2>(uv.Payload(*h))[0].x += 10.f;
            }
            if (delimiter == Sharp) {
                std::vector<uint32_t> selected;
                for (const auto edge : original.edges())
                    if (original.GetOppositeHalfedge(original.GetHalfedge(edge, 0u))) selected.push_back(*edge - original.EdgeFirst());
                f.SelectElements(entity, selected, Element::Edge);
                f.Do(action::object::SetSelectedSharp{.Element = Element::Edge, .Sharp = true});
            }
            f.EnterEdit(Element::Face);
            f.Do(action::selection::SelectAll{});
            f.Do(action::mesh::Dissolve{.Mode = action::mesh::DissolveMode::Limited, .DelimitMaterials = enabled && delimiter == Material, .DelimitSharpEdges = enabled && delimiter == Sharp, .DelimitUVs = enabled && delimiter == UV});
            f.Checkpoint();
            ExpectCounts(f.ActiveMesh(), enabled ? Counts{4u, 5u, 2u, 2u} : Counts{4u, 4u, 1u, 2u});
            ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
        }
}

void TestDissolveControls() {
    for (const bool surface : {false, true})
        for (const bool enabled : {false, true}) {
            Fixture f{"dissolve-controls"};
            MeshSource source;
            source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
            if (surface) {
                source.Data.AddFace(std::array{0u, 1u, 2u});
                source.Data.AddFace(std::array{0u, 2u, 3u});
            } else source.Data.Edges = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
            const auto id = CreateMesh(f.R, std::move(source)).StoreId;
            const auto [entity, instance] = f.AddEditable(id, surface ? Element::Edge : Element::Vertex);
            if (surface) {
                const auto mesh = f.ActiveMesh();
                for (const auto edge : mesh.edges())
                    if (mesh.GetOppositeHalfedge(mesh.GetHalfedge(edge, 0u)))
                        f.SelectElements(entity, std::array{*edge - mesh.EdgeFirst()}, Element::Edge);
            } else f.Do(action::selection::SelectAll{});
            const auto before = Positions(f.ActiveMesh());
            f.Do(action::mesh::Dissolve{.Mode = surface ? action::mesh::DissolveMode::Edges : action::mesh::DissolveMode::Limited, .Angle = 0.f, .KeepVertices = surface && enabled, .AllBoundaries = !surface && enabled});
            f.Checkpoint();
            ExpectCounts(f.ActiveMesh(), surface && enabled ? Counts{4u, 4u, 1u, 2u} : !surface && !enabled ? Counts{4u, 3u, 0u, 0u} :
                                                                                                              Counts{2u, 1u, 0u, 0u});
            if (f.ActiveMesh().VertexCount() == 2u) ExpectPositions(f.ActiveMesh(), {before[surface ? 1u : 0u], before[3]});
        }
}

void TestWeldDuplicateFaces() {
    for (const bool reverse : {false, true}) {
        Fixture f{"weld-duplicate-faces"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u});
        source.Data.AddFace(reverse ? std::array{5u, 4u, 3u} : std::array{3u, 4u, 5u});
        source.Attrs.TexCoords0 = std::vector<vec2>{{0, 0}, {1, 0}, {0, 1}, {9, 9}, {9, 9}, {9, 9}};
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        f.AddEditable(id, Element::Vertex);
        f.Do(action::selection::SelectAll{});
        f.Do(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = .001f});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectCounts(mesh, {3u, 3u, 1u, 1u});
        const auto &store = f.R.Context.get<const MeshStore>();
        for (const auto face : mesh.faces())
            for (const auto h : mesh.fh_range(face)) {
                const auto position = mesh.GetPosition(mesh.GetToVertex(h));
                expect(store.Arenas().CornerUvs[0].Get(*h) == vec2{position.x, position.y});
            }
        ExpectPolygonTessellation(store, mesh);
    }
}

void TestWeldDistanceGroups() {
    for (uint32_t kind = 0u; kind < 2u; ++kind) {
        Fixture f{"weld-distance-groups"};
        MeshSource source;
        source.Data.Positions = kind == 0u ? std::vector<vec3>{{0, 0, 0}, {2, -1, 0}, {4, 0, 0}, {2, 1, 0}, {.001f, 0, 0}, {2, 2, 0}, {4.001f, 0, 0}, {2, -2, 0}} :
                                             std::vector<vec3>{{-2, 0, 0}, {0, -2, 0}, {2, 0, 0}, {0, -1.999f, 0}, {1, -1, 0}, {2, -1, 0}, {2, .001f, 0}, {0, 2, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u});
        auto &uvs = source.Attrs.TexCoords0.emplace();
        for (uint32_t i = 0u; i < 8u; ++i) uvs.push_back({float(i), 0.f});
        const auto positions = source.Data.Positions;
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        f.SelectElements(entity, kind == 0u ? std::vector<uint32_t>{0u, 2u, 4u, 6u} : std::vector<uint32_t>{1u, 2u, 3u, 6u}, Element::Vertex);
        const auto first = f.ActiveMesh().VertexFirst();
        f.Do(action::mesh::Merge{.Mode = action::mesh::MergeMode::ByDistance, .Distance = .02f});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        const auto &meshes = f.R.Context.get<const MeshStore>();
        ExpectCounts(mesh, {6u, kind == 0u ? 8u : 7u, kind == 0u ? 2u : 1u, 4u});
        const auto targets = kind == 0u ? std::array{0u, 1u, 2u, 3u, 0u, 5u, 2u, 7u} : std::array{0u, 1u, 2u, 1u, 4u, 5u, 2u, 7u};
        for (const auto vertex : mesh.vertices()) {
            const auto index = *vertex - first;
            expect(index < targets.size() && targets[index] == index);
            if (index < positions.size()) expect(Near(mesh.GetPosition(vertex), positions[index]));
        }
        std::set<std::vector<uint32_t>> actual;
        for (const auto face : mesh.faces()) {
            std::vector<uint32_t> sources;
            for (const auto h : mesh.fh_range(face)) {
                const auto uv = meshes.Arenas().CornerUvs[0].Get(*h);
                const auto index = uint32_t(std::lround(uv.x));
                expect(index < targets.size() && uv == vec2{float(index), 0.f});
                if (index < targets.size()) expect(*mesh.GetToVertex(h) - first == targets[index]);
                sources.push_back(index);
            }
            std::rotate(sources.begin(), std::ranges::min_element(sources), sources.end());
            actual.insert(std::move(sources));
        }
        const std::set<std::vector<uint32_t>> expected = kind == 0u ? std::set<std::vector<uint32_t>>{{0u, 1u, 2u, 3u}, {4u, 5u, 6u, 7u}} :
                                                                      std::set<std::vector<uint32_t>>{{0u, 3u, 4u, 5u, 6u, 7u}};
        expect(actual == expected);
        std::set<uint64_t> wires;
        for (const auto edge : mesh.edges()) {
            const auto h = mesh.GetHalfedge(edge, 0u);
            if (!mesh.GetConnectivity().FaceOf(h)) wires.insert(MeshEdgeUsers::Key(*mesh.GetFromVertex(h) - first, *mesh.GetToVertex(h) - first));
        }
        expect(wires == (kind == 0u ? std::set<uint64_t>{} : std::set{MeshEdgeUsers::Key(1u, 2u)}));
        ExpectPolygonTessellation(meshes, mesh);
    }
}

void TestConcaveFillAndDissolve() {
    Fixture f{"concave-fill-dissolve"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {3, 4, 0}, {3, 1, 0}, {1, 1, 0}, {1, 4, 0}, {0, 4, 0}};
    for (uint32_t i = 0u; i < 8u; ++i) source.Data.Edges.push_back({i, (i + 1u) % 8u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Vertex);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Fill{});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 8u, 1u, 6u});
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    f.EnterEdit(Element::Face);
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Triangulate{});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 13u, 6u, 6u});
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    f.Do(action::mesh::Dissolve{action::mesh::DissolveMode::Faces});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 8u, 1u, 6u});
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    f.P->Undo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 13u, 6u, 6u});
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
    f.P->Redo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {8u, 8u, 1u, 6u});
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), f.ActiveMesh());
}

void TestSplitConcaveFaces() {
    Fixture f{"split-concave"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {4, 0, 0}, {4, 4, 0}, {3, 4, 0}, {3, 1, 0}, {1, 1, 0}, {1, 4, 0}, {0, 4, 0}, {10, 0, 0}, {13, 0, 0}, {13, 1, 0}, {11, 1, 0}, {11, 3, 0}, {10, 3, 0}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u});
    source.Data.AddFace(std::array{8u, 9u, 10u, 11u, 12u, 13u});
    auto &uvs = source.Attrs.TexCoords0.emplace();
    for (const auto p : source.Data.Positions) uvs.push_back({p.x, p.y});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Face);
    f.SelectElements(entity, std::array{0u}, Element::Face);
    f.P->History.Commit("Select concave face", {});
    const auto original = Positions(f.ActiveMesh());
    const auto untouched = f.ActiveMesh().FaceAt(1u);
    const auto untouched_points = FacePositions(f.ActiveMesh(), untouched);
    f.Do(action::mesh::SplitConcaveFaces{});
    f.Checkpoint();
    const auto check = [&](bool reversed) {
        const auto mesh = f.ActiveMesh();
        ExpectPositions(mesh, original);
        ExpectCounts(mesh, {14u, 16u, 4u, 10u});
        expect(FacePositions(mesh, untouched) == untouched_points);
        expect(f.Selection(Element::Face).Count() == 3u);
        std::set<std::vector<uint32_t>> actual;
        for (const auto face : mesh.faces()) {
            std::vector<uint32_t> loop;
            for (const auto h : mesh.fh_range(face)) {
                const auto vertex = mesh.GetToVertex(h);
                loop.push_back(uint32_t(std::ranges::find(original, vertex, &VertexPositions::value_type::first) - original.begin()));
                const auto p = mesh.GetPosition(vertex);
                expect(f.R.Context.get<const MeshStore>().Arenas().CornerUvs[0].Get(*h) == vec2{p.x, p.y});
            }
            std::rotate(loop.begin(), std::ranges::min_element(loop), loop.end());
            if (reversed && face != untouched) std::reverse(loop.begin() + 1, loop.end());
            actual.insert(std::move(loop));
        }
        // bmesh.ops.connect_verts_concave partitions, Blender 5.2.2; retain winding and source corners.
        expect(actual == std::set<std::vector<uint32_t>>{{0u, 5u, 6u, 7u}, {0u, 1u, 4u, 5u}, {1u, 2u, 3u, 4u}, {8u, 9u, 10u, 11u, 12u, 13u}});
        ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), mesh);
    };
    check(false);
    const auto split = *f.P->History.Present;
    f.Do(action::mesh::SplitConcaveFaces{});
    f.Checkpoint();
    expect(*f.P->History.Present == split) << "convex faces should not create history";
    f.P->Undo();
    f.Do(action::mesh::FlipNormals{});
    f.Do(action::mesh::SplitConcaveFaces{});
    f.Checkpoint();
    check(true);
}

void TestShrinkFatten() {
    // bpy.ops.transform.shrink_fatten(value=.2), Blender 5.2.2 d13f752e3b9c.
    // The unselected third face changes full vertex normals but not face-mode normals.
    const std::array<std::array<vec3, 6>, 4> references{{
        {{{0.f, .141421363f, .141421363f}, {2.f, .141421363f, .141421363f}, {0.f, 1.f, .2f}, {0.f, .2f, 1.f}, {-1.f, 2.f, 1.f}, {2.f, 2.f, 2.f}}},
        {{{0.f, .2f, .2f}, {2.f, .2f, .2f}, {0.f, 1.f, .2f}, {0.f, .2f, 1.f}, {-1.f, 2.f, 1.f}, {2.f, 2.f, 2.f}}},
        {{{.033676531f, .121543162f, .155219704f}, {2.f, .141421363f, .141421363f}, {.100692429f, 1.f, .172803462f}, {0.f, 0.f, 1.f}, {-1.f, 2.f, 1.f}, {2.f, 2.f, 2.f}}},
        {{{.043392085f, .156607896f, .2f}, {2.f, .2f, .2f}, {.116539836f, 1.f, .2f}, {0.f, 0.f, 1.f}, {-1.f, 2.f, 1.f}, {2.f, 2.f, 2.f}}},
    }};
    for (const auto mode : {Element::Face, Element::Vertex})
        for (const bool even : {false, true}) {
            Fixture f{"shrink-fatten"};
            MeshSource source;
            source.Data.Positions = {{0.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {-1.f, 2.f, 1.f}, {2.f, 2.f, 2.f}};
            source.Data.AddFace(std::array{0u, 1u, 2u});
            source.Data.AddFace(std::array{1u, 0u, 3u});
            source.Data.AddFace(std::array{0u, 2u, 4u});
            const auto id = CreateMesh(f.R, std::move(source)).StoreId;
            const auto [entity, instance] = f.AddEditable(id, mode);
            const std::vector<uint32_t> selected = mode == Element::Face ? std::vector{0u, 1u} : std::vector{0u, 1u, 2u};
            f.SelectElements(entity, selected, mode);
            f.P->History.Commit("Select offset reference", {});
            const auto counts = CountsOf(f.ActiveMesh());
            f.Do(action::mesh::ShrinkFatten{.2f, even});
            f.Checkpoint();
            const auto mesh = f.ActiveMesh();
            ExpectCounts(mesh, counts);
            const auto &reference = references[2u * (mode == Element::Vertex) + uint32_t(even)];
            for (uint32_t i = 0u; i < reference.size(); ++i)
                expect(Length(mesh.GetPosition(mesh.VertexAt(i)) - reference[i]) < .0002f) << "Blender normal offset" << i;
            expect(f.Selection(mode).Count() == selected.size());
        }
}

void TestShrinkFattenNonmanifold() {
    Fixture f{"shrink-fatten-nonmanifold"};
    MeshSource source;
    source.Data.Positions = {{0.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {-1.f, 2.f, 1.f}};
    source.Data.AddFace(std::array{0u, 1u, 2u});
    source.Data.AddFace(std::array{1u, 0u, 3u});
    source.Data.AddFace(std::array{0u, 1u, 4u});
    // A disconnected, opposite-winding pair must keep its cancelling normals fixed.
    source.Data.Positions.insert(source.Data.Positions.end(), {{10, 0, 0}, {11, 0, 0}, {10, 1, 0}});
    source.Data.AddFace(std::array{5u, 6u, 7u});
    source.Data.AddFace(std::array{7u, 6u, 5u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    f.AddEditable(id, Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    // Blender 5.2.2: all three users of the nonmanifold edge contribute to the normal and shell factor.
    const std::array<std::array<vec3, 5>, 2> reference{{
        {{{0.f, .039776370f, .196004689f}, {2.f, .033695214f, .197141156f}, {0.f, 1.f, .2f}, {0.f, .2f, 1.f}, {-1.f, 1.91055727f, 1.17888546f}}},
        {{{0.f, .093217827f, .459346354f}, {2.f, .086327799f, .505079567f}, {0.f, 1.f, .2f}, {0.f, .2f, 1.f}, {-1.f, 1.91055727f, 1.17888546f}}},
    }};
    for (const bool even : {false, true}) {
        f.Stage(action::mesh::ShrinkFatten{.2f, even});
        f.Checkpoint();
        for (uint32_t i = 0u; i < original.size(); ++i)
            expect(Length(f.ActiveMesh().GetPosition(original[i].first) - (i < 5u ? reference[uint32_t(even)][i] : original[i].second)) < .0002f);
        f.Cancel();
        ExpectPositions(f.ActiveMesh(), original);
    }
}

void TestRadialEdits() {
    Fixture f{"radial-edits"};
    MeshSource source;
    source.Data.Positions = {{-2.f, 0.f, 0.f}, {2.f, 0.f, 0.f}, {0.f, -1.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 0.f}, {9.f, 8.f, 7.f}};
    // Cross a reduction tile without changing the selected center or total radius.
    source.Data.Positions.insert(source.Data.Positions.begin() + 5, 252u, vec3{});
    source.Data.AddFace(std::array{0u, 2u, 1u, 3u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    std::array<uint32_t, 257> selected;
    std::iota(selected.begin(), selected.end(), 0u);
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select radial edit vertices", {});
    const auto original = Positions(f.ActiveMesh());
    const auto counts = CountsOf(f.ActiveMesh());
    const auto check = [&](const state::Scene &, bool sphere, float factor) {
        auto expected = original;
        for (uint32_t i = 0u; i < 4u; ++i) {
            const float radius = Length(original[i].second);
            const float target = sphere ? radius * (1.f - factor) + (6.f / float(selected.size())) * factor : radius - factor;
            expected[i].second = original[i].second * (target / radius);
        }
        ExpectPositions(f.ActiveMesh(), expected);
        ExpectCounts(f.ActiveMesh(), counts);
        expect(f.Selection(Element::Vertex).Count() == selected.size());
    };
    f.Stage(action::mesh::ToSphere{.25f});
    f.Checkpoint();
    check(f.R, true, .25f);
    f.Cancel();
    f.Stage(action::mesh::PushPull{-.4f});
    f.Checkpoint();
    check(f.R, false, -.4f);
    f.Cancel();
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::ToSphere{0.f});
    f.Do(action::mesh::ToSphere{-1.f});
    f.Do(action::mesh::PushPull{0.f});
    expect(*f.P->History.Present == history);
    const std::array center{4u};
    f.SelectElements(entity, center, Element::Vertex);
    f.Do(action::mesh::ToSphere{});
    f.Do(action::mesh::PushPull{});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), original);
}

void TestRadialTransformedMeshes() {
    Fixture f{"radial-transformed"};
    const std::array<std::array<vec3, 3>, 2> positions{{
        {{{-2.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}}},
        {{{0.f, -1.f, 0.f}, {2.f, 1.f, 0.f}, {0.f, 0.f, 2.f}}},
    }};
    // Blender 5.2.2 with center_override=(2/3,1/6,1/3), the shared vertex median in world space.
    // Background Blender otherwise uses its bounding-box pivot, independent of the scene setting.
    const std::array<std::array<vec3, 3>, 2> sphere{{
        {{{-1.602138877f, 0.254631042f, 0.063657761f}, {1.016452074f, -0.023930311f, -0.005982578f}, {-0.127709180f, 0.489163578f, -0.510836482f}}},
        {{{0.078688025f, -0.965901852f, 0.015737623f}, {2.049850702f, 1.191094637f, -0.049850762f}, {0.283103585f, -0.047183931f, 1.716896415f}}},
    }};
    const std::array<std::array<vec3, 3>, 2> push{{
        {{{-1.666109681f, 0.213689774f, 0.053422447f}, {0.778049350f, 0.322837293f, 0.080709331f}, {0.069631100f, 1.278524160f, 0.278524280f}}},
        {{{0.360994071f, -0.843569219f, 0.072198816f}, {1.902101994f, 0.624724150f, 0.097898051f}, {0.280898780f, -0.046816461f, 1.719101191f}}},
    }};
    std::array<uint32_t, 2> ids;
    std::array<state::Entity, 2> instances;
    std::array<VertexPositions, 2> original;
    for (uint32_t m = 0u; m < 2u; ++m) {
        MeshSource source;
        source.Data.Positions.assign(positions[m].begin(), positions[m].end());
        ids[m] = CreateMesh(f.R, std::move(source)).StoreId;
        const Transform transform = m == 0u ? Transform{.P = {2.f, 0.f, 0.f}, .R = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0.f, 0.f, 1.f}), .S = {2.f, 1.f, 1.f}} :
                                              Transform{.P = {-1.f, 1.f, 0.f}, .S = {1.f, 3.f, 1.f}};
        instances[m] = AddMesh(f.R, ids[m], MeshInstanceCreateInfo{.Transform = transform}).second;
        original[m] = Positions(f.MeshOf(ids[m]));
    }
    f.P->Settle();
    f.Do(action::selection::Select{instances[0]});
    f.Do(action::selection::ToggleSelected{instances[1]});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    for (const bool to_sphere : {true, false}) {
        if (to_sphere) f.Stage(action::mesh::ToSphere{.6f});
        else f.Stage(action::mesh::PushPull{.4f});
        f.Checkpoint();
        for (uint32_t m = 0u; m < 2u; ++m) {
            auto expected = original[m];
            for (uint32_t i = 0u; i < 3u; ++i) expected[i].second = (to_sphere ? sphere : push)[m][i];
            ExpectPositions(f.MeshOf(ids[m]), expected);
        }
        f.Cancel();
        f.Checkpoint();
        for (uint32_t m = 0u; m < 2u; ++m) ExpectPositions(f.MeshOf(ids[m]), original[m]);
    }
}

void TestShearTransformedMeshes() {
    Fixture f{"shear-transformed"};
    const std::array<std::array<vec3, 3>, 2> positions{{
        {{{-2, 0, 0}, {1, 0, 0}, {0, 1, 0}}},
        {{{0, -1, 0}, {2, 1, 0}, {0, 0, 2}}},
    }};
    // Blender 5.2.2, active object 0, angle=.6, Z/X axes, shared median.
    const std::array<std::array<std::array<vec3, 3>, 2>, 2> reference = {{
        {{{{{-2, -2.850570202f, 0}, {1, 1.254251003f, 0}, {0, .885977149f, 0}}},
          {{{-1.482296467f, -1, 0}, {4.622524738f, 1, 0}, {.570114136f, 0, 2}}}}},
        {{{{{-2.684136868f, 0, 0}, {.315863132f, 0, 0}, {-.342068434f, 1, 0}}},
          {{{0, -.771954298f, 0}, {2, 1.684136748f, 0}, {0, .228045583f, 2}}}}},
    }};
    std::array<uint32_t, 2> ids;
    std::array<state::Entity, 2> instances;
    std::array<VertexPositions, 2> original;
    for (uint32_t m = 0u; m < 2u; ++m) {
        MeshSource source;
        source.Data.Positions.assign(positions[m].begin(), positions[m].end());
        ids[m] = CreateMesh(f.R, std::move(source)).StoreId;
        const Transform transform = m == 0u ? Transform{.P = {2, 0, 0}, .R = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}), .S = {2, 1, -1.f}} :
                                              Transform{.P = {-1, 1, 0}, .S = {-1.f, 3, .5f}};
        instances[m] = AddMesh(f.R, ids[m], MeshInstanceCreateInfo{.Transform = transform}).second;
        original[m] = Positions(f.MeshOf(ids[m]));
    }
    f.P->Settle();
    f.Do(action::selection::Select{instances[0]});
    f.Do(action::selection::ToggleSelected{instances[1]});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::Shear{.6f, action::mesh::ShearAxis::X, action::mesh::ShearAxis::X});
    f.Do(action::mesh::Shear{std::numeric_limits<float>::quiet_NaN()});
    expect(*f.P->History.Present == history);
    for (const bool local : {false, true}) {
        f.Stage(action::mesh::Shear{.Angle = .6f, .Local = local});
        f.Checkpoint();
        for (uint32_t m = 0u; m < 2u; ++m) {
            auto expected = original[m];
            for (uint32_t i = 0u; i < 3u; ++i) expected[i].second = reference[uint32_t(local)][m][i];
            ExpectPositions(f.MeshOf(ids[m]), expected);
        }
        f.Cancel();
        f.Checkpoint();
        for (uint32_t m = 0u; m < 2u; ++m) ExpectPositions(f.MeshOf(ids[m]), original[m]);
    }
}

std::vector<uint32_t> EdgePairOrdinals(const Mesh &mesh, std::span<const std::array<uint32_t, 2>> pairs) {
    std::set<uint64_t> keys;
    for (const auto pair : pairs) keys.insert(MeshEdgeUsers::Key(pair[0], pair[1]));
    std::vector<uint32_t> selected;
    uint32_t ordinal = 0u;
    for (const auto e : mesh.edges()) {
        const auto h = mesh.GetHalfedge(e, 0u);
        const auto a = mesh.VertexOrdinal(mesh.GetFromVertex(h)), b = mesh.VertexOrdinal(mesh.GetToVertex(h));
        if (keys.contains(MeshEdgeUsers::Key(a, b))) selected.push_back(ordinal);
        ++ordinal;
    }
    return selected;
}

void SelectEdgePairs(Fixture &f, state::Entity entity, std::span<const std::array<uint32_t, 2>> pairs) {
    const auto selected = EdgePairOrdinals(f.ActiveMesh(), pairs);
    expect(selected.size() == pairs.size());
    f.SelectElements(entity, selected, Element::Edge);
    f.P->History.Commit("Select slide edges", {});
}

void TestEdgeSlideReferences() {
    // Blender 5.2.2 edge_slide, with its positive side pointing up in this fixture.
    const std::array<std::array<vec3, 3>, 5> reference{{
        {{{-.125f, .5f, 0}, {2, .75f, 0}, {4.25f, .25f, 0}}},
        {{{-.05f, -.25f, 0}, {2, -.5f, 0}, {4.125f, -.25f, 0}}},
        {{{-.22286583f, .89146334f, 0}, {2, -.11564839f, 0}, {4.85387611f, .85387623f, 0}}},
        {{{.125f, -.5f, 0}, {2, -.75f, 0}, {3.75f, -.25f, 0}}},
        {{{-.1304878f, .5219512f, 0}, {2, 1.49251878f, 0}, {4.17364359f, -.34728706f, 0}}},
    }};
    for (uint32_t sample = 0u; sample < reference.size(); ++sample) {
        const auto kind = std::array{0u, 1u, 3u, 6u, 8u}[sample];
        Fixture f{"edge-slide-reference"};
        MeshSource source;
        source.Data.Positions = {{-.2f, -1, 0}, {2, -2, 0}, {4.5f, -1, 0}, {0, 0, 0}, {2, 0, 0}, {4, 0, 0}, {-.5f, 2, 0}, {2, 3, 0}, {5, 1, 0}};
        source.Data.AddFace(std::array{0u, 1u, 4u, 3u});
        source.Data.AddFace(std::array{1u, 2u, 5u, 4u});
        source.Data.AddFace(std::array{3u, 4u, 7u, 6u});
        source.Data.AddFace(std::array{4u, 5u, 8u, 7u});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Edge);
        const std::array pairs{std::array{3u, 4u}, std::array{4u, 5u}};
        SelectEdgePairs(f, entity, pairs);
        const auto original = Positions(f.ActiveMesh());
        const auto counts = CountsOf(f.ActiveMesh());
        auto expected = original;
        for (uint32_t i = 0u; i < 3u; ++i) expected[i + 3u].second = reference[sample][i];
        const action::mesh::EdgeSlide action{.Factor = kind == 1u || kind == 6u ? -.25f : kind == 8u ? 0.f :
                                                                                                       .25f,
                                             .Direction = {0, 1, 0},
                                             .Even = (kind >= 2u && kind <= 5u) || kind == 8u,
                                             .Flipped = kind == 3u || kind == 5u,
                                             .Clamp = kind < 4u || kind == 8u};
        f.Do(action);
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), expected);
        ExpectCounts(f.ActiveMesh(), counts);
        expect(f.Selection(Element::Edge).Count() == 2u);
    }
}

void TestEdgeSlideFacePaths() {
    // Blender 5.2.2: forked rails in triangle strips, a pentagon, a quad turn,
    // and nonplanar fork intersections. Only the immediate face paths move.
    const std::array<std::array<vec3, 3>, 4> reference{{
        {{{.5f, .75f, 0}, {2.375f, .5f, 0}, {4.25f, .25f, 0}}},
        {{{0, .5f, 0}, {1.72951460f, .65300971f, 0}, {2.625f, 1.5f, 0}}},
        {{{0, .5f, 0}, {1.5f, .5f, 0}, {2.25f, 2, 0}}},
        {{{.5f, .75f, .15f}, {2.375f, .5f, .0375f}, {4.25f, .25f, -.075f}}},
    }};
    for (uint32_t kind = 0u; kind < 4u; ++kind) {
        Fixture f{"edge-slide-face-paths"};
        MeshSource source;
        if (kind == 0u || kind == 3u) {
            source.Data.Positions = {{-.2f, -1, 0}, {2, -2, 0}, {4.5f, -1, 0}, {0, 0, 0}, {2, 0, 0}, {4, 0, 0}, {-.5f, 2, 0}, {2, 3, 0}, {5, 1, 0}};
            if (kind == 3u) {
                const std::array heights{.4f, -.2f, .5f, 0.f, 0.f, 0.f, .3f, .6f, -.3f};
                for (uint32_t i = 0u; i < heights.size(); ++i) source.Data.Positions[i].z = heights[i];
            }
            for (const auto face : std::array{std::array{0u, 1u, 4u}, std::array{0u, 4u, 3u}, std::array{1u, 2u, 5u}, std::array{1u, 5u, 4u}, std::array{3u, 4u, 7u}, std::array{3u, 7u, 6u}, std::array{4u, 5u, 8u}, std::array{4u, 8u, 7u}}) source.Data.AddFace(face);
        } else {
            source.Data.Positions = kind == 1u ? std::vector<vec3>{{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {1.5f, 3, 0}, {0, 2, 0}} :
                                                 std::vector<vec3>{{0, 0, 0}, {2, 0, 0}, {3, 2, 0}, {0, 2, 0}};
            std::vector<uint32_t> face;
            for (uint32_t i = 0u; i < source.Data.Positions.size(); ++i) face.push_back(i);
            source.Data.AddFace(face);
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const Transform world = kind == 3u ? Transform{.P = {2, -1, 3}, .R = AngleAxis(.6f, Normalize(vec3{1, 2, 3})), .S = {3, -2, .5f}} : Transform{};
        const auto [entity, instance] = f.AddEditable(id, Element::Edge, MeshInstanceCreateInfo{.Transform = world});
        const uint32_t first = kind == 0u || kind == 3u ? 3u : 0u;
        const std::array pairs{std::array{first, first + 1u}, std::array{first + 1u, first + 2u}};
        SelectEdgePairs(f, entity, pairs);
        const auto original = Positions(f.ActiveMesh());
        auto expected = original;
        for (uint32_t i = 0u; i < 3u; ++i) expected[first + i].second = reference[kind][i];
        f.Stage(action::mesh::EdgeSlide{.Factor = .25f, .Direction = world.R * vec3{0, world.S.y < 0.f ? -1.f : 1.f, 0}});
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), expected);
        f.Commit();
    }
    Fixture f{"edge-slide-nonmanifold"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}};
    source.Data.AddFace(std::array{0u, 1u, 2u});
    source.Data.AddFace(std::array{1u, 0u, 3u});
    source.Data.AddFace(std::array{0u, 1u, 4u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Edge);
    // Nonmanifold inputs may have multiple canonical representatives. Select one.
    const auto edges = EdgePairOrdinals(f.ActiveMesh(), std::array<std::array<uint32_t, 2>, 1>{{{0u, 1u}}});
    expect(!edges.empty());
    const std::array selected{edges.front()};
    f.SelectElements(entity, selected, Element::Edge);
    f.P->History.Commit("Select nonmanifold slide edge", {});
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::EdgeSlide{});
    f.Checkpoint();
    expect(*f.P->History.Present == history);
}

void TestEdgeSlideClosedLoop() {
    Fixture f{"edge-slide-closed-loop"};
    MeshSource source;
    for (const float radius : {1.f, 2.f, 4.f}) {
        source.Data.Positions.push_back({radius, radius, 0});
        source.Data.Positions.push_back({-radius, radius, 0});
        source.Data.Positions.push_back({-radius, -radius, 0});
        source.Data.Positions.push_back({radius, -radius, 0});
    }
    for (uint32_t ring = 0u; ring < 2u; ++ring)
        for (uint32_t i = 0u; i < 4u; ++i)
            source.Data.AddFace(std::array{ring * 4u + i, ring * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + (i + 1u) % 4u, (ring + 1u) * 4u + i});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Edge);
    const std::array pairs{std::array{4u, 5u}, std::array{5u, 6u}, std::array{6u, 7u}, std::array{7u, 4u}};
    SelectEdgePairs(f, entity, pairs);
    const auto original = Positions(f.ActiveMesh());
    for (const bool even : {false, true}) {
        f.Stage(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {1, 1, 0}, .Even = even});
        f.Checkpoint();
        auto expected = original;
        for (uint32_t i = 4u; i < 8u; ++i) expected[i].second *= even ? 2.875f / 2.f : 2.5f / 2.f;
        ExpectPositions(f.ActiveMesh(), expected);
        f.Cancel();
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), original);
    }
    // Branching selections are invalid and must not create history.
    const std::array branch{std::array{4u, 5u}, std::array{4u, 7u}, std::array{4u, 8u}};
    SelectEdgePairs(f, entity, branch);
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::EdgeSlide{});
    f.Checkpoint();
    expect(*f.P->History.Present == history);
    ExpectPositions(f.ActiveMesh(), original);
}

void TestVertexSlideReferences() {
    // Blender 5.2.2 vert_slide, explicit world direction, correct_uv=false.
    // The first selected vertex is also Blender's cursor-nearest reference here.
    const std::array<std::array<vec3, 3>, 5> reference{{
        {{{.5f, 0, 0}, {1, 4, 0}, {.25f, 8, 0}}},
        {{{1.5f, 0, 0}, {3.5f, 4, 0}, {.5f, 8, 0}}},
        {{{2, 0, 0}, {4, 4, 0}, {1, 8, 0}}},
        {{{-1, 0, 0}, {-1, 4, 0}, {-1, 8, 0}}},
        {{{0, .5f, 0}, {-.223606795f, 4.447213650f, 0}, {0, 8.5f, 0}}},
    }};
    for (uint32_t sample = 0u; sample < reference.size(); ++sample) {
        const auto kind = std::array{0u, 2u, 3u, 5u, 7u}[sample];
        Fixture f{"vertex-slide-reference"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {0, 4, 0}, {4, 4, 0}, {-1, 6, 0}, {0, 8, 0}, {1, 8, 0}, {0, 10, 0}, {0, 12, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u});
        source.Data.AddFace(std::array{3u, 4u, 5u});
        source.Data.AddFace(std::array{6u, 7u, 8u});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto transform = kind >= 6u ? Transform{.P = {2, -1, 3}, .R = AngleAxis(.4f, vec3{0, 0, 1}) * AngleAxis(-.2f, vec3{0, 1, 0}) * AngleAxis(.3f, vec3{1, 0, 0}), .S = {-3, .5f, 2}} : Transform{};
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex, MeshInstanceCreateInfo{.Transform = transform});
        const std::array selected{0u, 3u, 6u, 9u};
        f.SelectElements(entity, selected, Element::Vertex);
        f.P->History.Commit("Select slide vertices", {});
        const auto original = Positions(f.ActiveMesh());
        auto expected = original;
        for (uint32_t i = 0u; i < 3u; ++i) expected[3u * i].second = reference[sample][i];
        const action::mesh::VertexSlide action{.Factor = kind == 3u ? 0.f : kind == 4u ? 1.5f :
                                                   kind == 5u                          ? -.5f :
                                                                                         .25f,
                                               .Direction = kind >= 6u ? vec3{-.3f, 1, .2f} : vec3{1, 0, 0},
                                               .Even = (kind >= 1u && kind <= 5u) || kind == 7u,
                                               .Flipped = kind == 2u || kind == 3u,
                                               .Clamp = kind != 5u};
        f.Do(action);
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), expected);
    }
}

void TestSlideAttributes() {
    for (const bool edge : {false, true}) {
        Fixture f{"slide-preserves-attributes"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {2, -2, 0}, {8, 0, 0}, {10, 0, 0}, {8, 2, 0}};
        for (const auto face : std::array{std::array{0u, 1u, 2u}, std::array{1u, 0u, 3u}, std::array{4u, 5u, 6u}}) source.Data.AddFace(face);
        source.Attrs.TexCoords0 = std::vector<vec2>(7u);
        source.Attrs.Colors0 = std::vector<vec4>(7u);
        source.Attrs.Colors0ComponentCount = 4u;
        source.Attrs.Tangents = std::vector<vec4>(7u, vec4{1, 0, 0, 1});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto &a = f.R.Context.get<const MeshStore>().Arenas();
        // Different values on every corner retain an explicit UV/color seam at the shared edge.
        for (const auto face : f.MeshOf(id).faces())
            for (const auto h : f.MeshOf(id).fh_range(face)) {
                a.CornerUvs[0].Values.Buffer.GetMutableSpan<vec2>(a.CornerUvs[0].Payload(*h))[0] = {float(*h), float(*face)};
                a.CornerColors.Values.Buffer.GetMutableSpan<vec4>(a.CornerColors.Payload(*h))[0] = {float(*h) / 16.f, 0, 1, 1};
            }
        const auto [entity, instance] = f.AddEditable(id, edge ? Element::Edge : Element::Vertex);
        if (edge) SelectEdgePairs(f, entity, std::array<std::array<uint32_t, 2>, 1>{{{0u, 1u}}});
        else {
            f.SelectElements(entity, std::array{0u}, Element::Vertex);
            f.P->History.Commit("Select attributed slide", {});
        }
        const auto before = Positions(f.ActiveMesh());
        const auto check = [&](const state::Scene &, bool edited) {
            const auto mesh = f.ActiveMesh();
            ExpectCounts(mesh, {7u, 8u, 3u, 3u});
            expect((Positions(mesh) != before) == edited);
            for (const auto face : mesh.faces())
                for (const auto h : mesh.fh_range(face)) {
                    expect(a.CornerUvs[0].Get(*h) == vec2{float(*h), float(*face)});
                    expect(a.CornerColors.Get(*h) == vec4{float(*h) / 16.f, 0, 1, 1});
                    expect(a.CornerTangents.Get(*h) == (edited && face != mesh.FaceAt(2u) ? vec4{} : vec4{1, 0, 0, 1}));
                }
        };
        f.Stage(action::view::TransformElements{});
        f.Commit();
        f.Checkpoint();
        check(f.R, false);
        if (edge) f.Stage(action::mesh::EdgeSlide{.Factor = .25f, .Direction = {0, 1, 0}});
        else f.Stage(action::mesh::VertexSlide{.Factor = .25f});
        f.Commit();
        f.Checkpoint();
        check(f.R, true);
        f.CheckUndoRedo(check);
    }
}

void TestVertexSlideReferenceAndDegeneracy() {
    Fixture f{"vertex-slide-reference-choice"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {0, 3, 0}, {4, 3, 0}, {7, 0, 0}, {7, 0, 0}, {9, 0, 0}, {0, 9, 0}};
    source.Data.Edges = {{0u, 1u}, {2u, 3u}, {4u, 5u}, {4u, 6u}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    const std::array selected{0u, 2u, 7u};
    ApplyEditSelectionLists(f.R, std::array{std::pair{entity, std::span<const uint32_t>{selected}}}, Element::Vertex);
    f.R.emplace_or_replace<MeshActiveElement>(entity, *f.ActiveMesh().VertexAt(2u));
    f.P->Settle();
    f.P->History.Commit("Select slide reference", {});
    const auto original = Positions(f.ActiveMesh());
    auto expected = original;
    expected[0].second.x += 1.f;
    expected[2].second.x += 1.f;
    f.Stage(action::mesh::VertexSlide{.Factor = .25f, .Even = true});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected);
    f.Cancel();
    const std::array pair{0u, 1u};
    f.SelectElements(entity, pair, Element::Vertex);
    f.P->History.Commit("Select both slide endpoints", {});
    expected = original;
    expected[0].second.x = 1.f;
    expected[1].second.x = 1.f;
    f.Stage(action::mesh::VertexSlide{});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected);
    f.Cancel();
    const std::array degenerate{4u, 7u};
    f.SelectElements(entity, degenerate, Element::Vertex);
    f.P->History.Commit("Select degenerate slide", {});
    for (const bool even : {false, true}) {
        f.Stage(action::mesh::VertexSlide{.Direction = {-1, 0, 0}, .Even = even, .Flipped = true});
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), original);
        f.Cancel();
    }
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::VertexSlide{.Direction = {}});
    f.Do(action::mesh::VertexSlide{.Factor = std::numeric_limits<float>::quiet_NaN()});
    expect(*f.P->History.Present == history);
}

void TestRandomizeControls() {
    Fixture f{"randomize-controls"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {9, 8, 7}};
    source.Data.AddFace(std::array{0u, 1u, 2u});
    const auto [entity, instance] = f.AddEditable(CreateMesh(f.R, std::move(source)).StoreId, Element::Vertex);
    const std::array selected{0u, 1u, 2u};
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select randomize vertices", {});
    const auto original = Positions(f.ActiveMesh());
    const auto sample = [&](action::mesh::Randomize action) {
        f.Stage(action);
        f.Checkpoint();
        auto result = Positions(f.ActiveMesh());
        f.Cancel();
        expect(result.back() == original.back());
        return result;
    };
    const auto unit = sample({.Amount = .25f, .Uniform = 1.f, .Seed = 3u});
    const auto repeat = sample({.Amount = .25f, .Uniform = 1.f, .Seed = 3u});
    const auto reseeded = sample({.Amount = .25f, .Uniform = 1.f, .Seed = 7u});
    const auto variable = sample({.Amount = .25f, .Seed = 3u});
    const auto normal = sample({.Amount = .25f, .Uniform = 1.f, .Normal = 1.f, .Seed = 3u});
    for (uint32_t i = 0u; i < selected.size(); ++i) {
        expect(std::abs(Length(unit[i].second - original[i].second) - .25f) < 1e-5f);
        expect(unit[i] == repeat[i]);
        expect(Length(unit[i].second - reseeded[i].second) > 1e-3f);
        expect(Length(variable[i].second - original[i].second) <= .25f);
        const auto delta = normal[i].second - original[i].second;
        expect(std::abs(delta.x) < 1e-5f && std::abs(delta.y) < 1e-5f && std::abs(std::abs(delta.z) - .25f) < 1e-5f);
    }
}

void TestBendReferences() {
    Fixture f{"bend-reference"};
    const std::array points{vec3{-1, 0, 0}, vec3{1, 0, 1}, vec3{2, 0, 2}, vec3{3, 0, 3}};
    const float root2 = std::sqrt(2.f);
    const std::array expected{vec3{-1, 0, 0}, vec3{root2, root2 - 2.f, 1}, vec3{2, -2, 2}, vec3{2, -3, 3}};
    const Transform world{.P = {3, -1, 2}, .R = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{0, 0, 1}), .S = {-2, .5f, 1.5f}};
    MeshSource source;
    for (const auto point : points) {
        const auto local = Conjugate(world.R) * (point - world.P);
        source.Data.Positions.push_back({local.x / world.S.x, local.y / world.S.y, local.z / world.S.z});
    }
    source.Data.Edges = {{0u, 1u}, {1u, 2u}, {2u, 3u}};
    f.AddEditable(CreateMesh(f.R, std::move(source)).StoreId, Element::Vertex, MeshInstanceCreateInfo{.Transform = world});
    f.Do(action::selection::SelectAll{});
    const auto original = Positions(f.ActiveMesh());
    f.Stage(action::mesh::Bend{.Angle = std::numbers::pi_v<float> / 2.f, .Radius = 2.f});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    for (uint32_t i = 0u; i < points.size(); ++i) {
        const auto p = mesh.GetPosition(mesh.VertexAt(i));
        expect(Near(world.P + world.R * vec3{world.S.x * p.x, world.S.y * p.y, world.S.z * p.z}, expected[i]));
    }
    f.Cancel();
    f.Stage(action::mesh::Bend{.Angle = 1e-6f, .Radius = 2.f, .Clamp = false});
    f.Checkpoint();
    for (const auto &[v, p] : original) expect(Length(f.ActiveMesh().GetPosition(v) - p) < .001f);
    f.Cancel();
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::Bend{.Radius = 0.f});
    expect(*f.P->History.Present == history);
}

void TestWarpReferences() {
    Fixture f{"warp-reference"};
    MeshSource source;
    // Span two reduction tiles; displacement is independent of the height.
    for (uint32_t z = 0u; z < 53u; ++z)
        for (const float x : {-3.f, -2.f, 0.f, 2.f, 3.f}) source.Data.Positions.push_back({x, 1.f, float(z)});
    f.AddEditable(CreateMesh(f.R, std::move(source)).StoreId, Element::Vertex);
    f.Do(action::selection::SelectAll{});
    const float root3 = std::sqrt(3.f) * .5f;
    // A positive half turn continues downward beyond both explicit bounds.
    const std::array<std::array<vec3, 5>, 2> references{{
        {{{-1, -1, 0}, {-1, 0, 0}, {0, 1, 0}, {1, 0, 0}, {1, -1, 0}}},
        {{{-1, 0, 0}, {-root3, .5f, 0}, {0, 1, 0}, {root3, .5f, 0}, {1, 0, 0}}},
    }};
    for (const bool automatic : {false, true}) {
        auto expected = Positions(f.ActiveMesh());
        for (uint32_t i = 0u; i < expected.size(); ++i) expected[i].second = references[automatic][i % 5u] + vec3{0, 0, float(i / 5u)};
        f.Stage(action::mesh::Warp{.Angle = std::numbers::pi_v<float>, .AutoRange = automatic, .Min = 2.f, .Max = -2.f});
        f.Checkpoint();
        ExpectPositions(f.ActiveMesh(), expected);
        f.Cancel();
    }
    // Unlike most transforms, Warp at angle zero collapses its horizontal range.
    f.Do(action::mesh::Warp{.Angle = 0.f});
    f.Checkpoint();
    for (const auto v : f.ActiveMesh().vertices()) expect(Near(f.ActiveMesh().GetPosition(v), vec3{0, 1, f.ActiveMesh().GetPosition(v).z}));
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::Warp{.AutoRange = false, .Min = 1.f, .Max = 1.f});
    expect(*f.P->History.Present == history);
}

void TestRecalculateNormals() {
    Fixture f{"recalculate-normals"};
    MeshSource source;
    source.Data.Positions = {{-1.f, -1.f, -1.f}, {1.f, -1.f, -1.f}, {1.f, 1.f, -1.f}, {-1.f, 1.f, -1.f}, {-1.f, -1.f, 1.f}, {1.f, -1.f, 1.f}, {1.f, 1.f, 1.f}, {-1.f, 1.f, 1.f}};
    std::array<std::array<uint32_t, 4>, 6> cube{{{3, 2, 1, 0}, {4, 5, 6, 7}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}}};
    for (const auto i : {0u, 2u, 5u}) std::ranges::reverse(cube[i]);
    for (const auto &face : cube) source.Data.AddFace(face);
    for (uint32_t i = 0u; i < 8u; ++i) source.Data.Positions.push_back(source.Data.Positions[i] + vec3{5.f, 1.f, 0.f});
    for (auto face : cube) {
        for (auto &v : face) v += 8u;
        source.Data.AddFace(face);
    }
    for (const auto p : std::array{vec3{10, 0, 0}, vec3{12, 0, 0}, vec3{10, 1, 0}, vec3{10, 0, 1}, vec3{10, -1, -1}, vec3{0, 4, 0}, vec3{1, 4, 0}, vec3{0, 5, 0}, vec3{0, 4, 1}, vec3{0, 8, 0}, vec3{1, 8, 0}, vec3{2, 8, 0}}) source.Data.Positions.push_back(p);
    for (const auto face : std::array{std::array{16u, 17u, 18u}, std::array{17u, 16u, 19u}, std::array{16u, 17u, 20u}, std::array{21u, 22u, 23u}, std::array{21u, 22u, 24u}, std::array{25u, 26u, 27u}}) source.Data.AddFace(face);
    auto &uvs = source.Attrs.TexCoords0.emplace();
    for (const auto p : source.Data.Positions) uvs.push_back({p.x, p.y});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Face);
    const std::array selected{0u, 1u, 2u, 3u, 4u, 5u, 6u, 12u, 13u, 14u, 15u, 16u, 17u};
    f.SelectElements(entity, selected, Element::Face);
    f.P->History.Commit("Select winding repair components", {});
    const auto original = Positions(f.ActiveMesh());
    const auto counts = CountsOf(f.ActiveMesh());
    std::vector<std::pair<he::FH, vec3>> normals;
    for (const auto face : f.ActiveMesh().faces()) normals.emplace_back(face, f.ActiveMesh().GetNormal(face));
    // bmesh.ops.recalc_face_normals, Blender 5.2.2 d13f752e3b9c, including the open and nonmanifold components.
    const std::array flipped{0u, 2u, 5u, 6u, 15u};
    const auto check = [&](const state::Scene &, bool changed, bool inside = false) {
        const auto mesh = f.ActiveMesh();
        ExpectPositions(mesh, original);
        ExpectCounts(mesh, counts);
        expect(f.Selection(Element::Face).Count() == selected.size());
        for (uint32_t i = 0u; i < normals.size(); ++i) {
            const bool reverse = changed && ((std::ranges::find(flipped, i) != flipped.end()) ^ (inside && std::ranges::find(selected, i) != selected.end()));
            expect(Near(mesh.GetNormal(normals[i].first), normals[i].second * (reverse ? -1.f : 1.f))) << "normal of face" << i;
        }
        const auto &a = f.R.Context.get<const MeshStore>().Arenas();
        for (const auto face : mesh.faces())
            for (const auto h : mesh.fh_range(face)) {
                const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                expect(a.CornerUvs[0].Get(*h) == vec2{p.x, p.y});
            }
    };
    f.Stage(action::mesh::RecalculateNormals{});
    f.Commit();
    f.Checkpoint();
    check(f.R, true);
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::RecalculateNormals{});
    f.Checkpoint();
    check(f.R, true);
    expect(*f.P->History.Present == history) << "consistent winding must not create history";
    f.Do(action::mesh::RecalculateNormals{true});
    f.Checkpoint();
    check(f.R, true, true);
}

void TestEditVisibility() {
    struct Reference {
        Element Mode;
        bool Unselected;
        std::vector<uint32_t> Vertices, Faces;
        std::vector<std::array<uint32_t, 2>> Edges;
    };
    // Blender 5.2.2 mesh.hide, including the unconnected vertex.
    const std::array references{
        Reference{Element::Vertex, false, {1}, {0, 1}, {{0, 1}, {1, 2}, {1, 4}}},
        Reference{Element::Vertex, true, {0, 2, 3, 4, 5, 6}, {0, 1}, {{0, 1}, {0, 3}, {1, 2}, {1, 4}, {2, 5}, {3, 4}, {4, 5}}},
        Reference{Element::Edge, false, {}, {0, 1}, {{1, 4}}},
        Reference{Element::Edge, true, {0, 2, 3, 5, 6}, {0, 1}, {{0, 1}, {0, 3}, {1, 2}, {2, 5}, {3, 4}, {4, 5}}},
        Reference{Element::Face, false, {0, 3}, {0}, {{0, 1}, {0, 3}, {3, 4}}},
        Reference{Element::Face, true, {2, 5, 6}, {1}, {{1, 2}, {2, 5}, {4, 5}}},
    };
    for (const auto &reference : references) {
        Fixture f{"edit-visibility"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {3, 1, 0}};
        source.Data.AddFace(std::array{0u, 1u, 4u, 3u});
        source.Data.AddFace(std::array{1u, 2u, 5u, 4u});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, reference.Mode);
        uint32_t selected = reference.Mode == Element::Vertex ? 1u : 0u;
        if (reference.Mode == Element::Edge)
            for (const auto e : f.ActiveMesh().edges()) {
                const auto h = f.ActiveMesh().GetHalfedge(e, 0u);
                const auto a = f.ActiveMesh().VertexOrdinal(f.ActiveMesh().GetFromVertex(h)), b = f.ActiveMesh().VertexOrdinal(f.ActiveMesh().GetToVertex(h));
                if ((a == 1u && b == 4u) || (a == 4u && b == 1u)) break;
                ++selected;
            }
        f.SelectElements(entity, {&selected, 1u}, reference.Mode);
        f.P->History.Commit("Select visibility fixture", {});
        const auto original = Positions(f.ActiveMesh());
        const auto counts = CountsOf(f.ActiveMesh());
        const auto check = [&] {
            const auto &meshes = f.R.Context.get<const MeshStore>();
            const auto mesh = f.ActiveMesh();
            std::vector<uint32_t> vertices, faces;
            std::vector<std::array<uint32_t, 2>> edges;
            meshes.GetHiddenElements(id, Element::Vertex).ForEach([&](uint32_t v) { vertices.push_back(mesh.VertexOrdinal(he::VH{v})); });
            meshes.GetHiddenElements(id, Element::Face).ForEach([&](uint32_t v) { faces.push_back(mesh.FaceOrdinal(he::FH{v})); });
            meshes.GetHiddenElements(id, Element::Edge).ForEach([&](uint32_t e) {
                const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
                std::array pair{mesh.VertexOrdinal(mesh.GetFromVertex(h)), mesh.VertexOrdinal(mesh.GetToVertex(h))};
                std::ranges::sort(pair);
                edges.push_back(pair);
            });
            std::ranges::sort(edges);
            expect(vertices == reference.Vertices) << "hidden vertices mode=" << uint32_t(reference.Mode) << " unselected=" << reference.Unselected;
            expect(faces == reference.Faces) << "hidden faces mode=" << uint32_t(reference.Mode) << " unselected=" << reference.Unselected;
            expect(edges == reference.Edges) << "hidden edges mode=" << uint32_t(reference.Mode) << " unselected=" << reference.Unselected;
            ExpectPositions(mesh, original);
            ExpectCounts(mesh, counts);
        };
        f.Do(action::mesh::Hide{reference.Unselected});
        f.Checkpoint();
        check();
        f.Do(action::selection::SelectAll{});
        f.Checkpoint();
        check();
        const auto visible = f.ActiveMesh().ElementCount(reference.Mode) - f.R.Context.get<const MeshStore>().GetHiddenElements(id, reference.Mode).Count();
        expect(f.Selection(reference.Mode).Count() == visible);
        const auto hidden_node = *f.P->History.Present;
        f.Do(action::mesh::Reveal{false});
        f.Checkpoint();
        expect(f.Selection(reference.Mode).Count() == visible);
        for (const auto mode : {Element::Vertex, Element::Edge, Element::Face}) expect(f.R.Context.get<const MeshStore>().GetHiddenElements(id, mode).Count() == 0u);
        // Reveal can match the hide node's parent, which history adopts directly.
        f.P->Navigate(hidden_node);
        f.Checkpoint();
        check();
        f.Do(action::mesh::Reveal{});
        f.Checkpoint();
        expect(f.Selection(reference.Mode).Count() == f.ActiveMesh().ElementCount(reference.Mode));
    }
}

void TestHiddenGeometryPicking() {
    for (const auto mode : {Element::Vertex, Element::Edge, Element::Face}) {
        Fixture f{"hidden-picking"};
        f.Checkpoint();
        MeshSource padding;
        padding.Data.Positions = {{10, 10, 10}, {11, 10, 10}, {10, 11, 10}};
        padding.Data.AddFace(std::array{0u, 1u, 2u});
        CreateMesh(f.R, std::move(padding));
        const auto normal = Normalize(f.FrameView().CameraPosition), right = Normalize(Cross(normal, vec3{0, 0, 1})), up = Cross(normal, right);
        MeshSource source;
        for (uint32_t layer = 0u; layer < 2u; ++layer) {
            const auto center = normal * (layer ? -.5f : .5f);
            if (mode == Element::Vertex) source.Data.Positions.push_back(center);
            else if (mode == Element::Edge) {
                source.Data.Positions.push_back(center - right);
                source.Data.Positions.push_back(center + right);
                source.Data.Edges.push_back({2u * layer, 2u * layer + 1u});
            } else {
                for (const auto offset : std::array{-right - up, right - up, right + up, -right + up}) source.Data.Positions.push_back(center + offset);
                source.Data.AddFace(std::array{4u * layer, 4u * layer + 1u, 4u * layer + 2u, 4u * layer + 3u});
            }
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, mode);
        const std::array selected{0u};
        f.SelectElements(entity, selected, mode);
        f.P->History.Commit("Select front geometry", {});
        f.Do(action::mesh::Hide{});
        f.Checkpoint();
        f.Click(f.ViewFraction(vec3{}));
        f.Checkpoint();
        expect(f.Selection(mode).Count() == 1u) << "visible element behind hidden geometry" << uint32_t(mode);
        const auto hidden = f.R.Context.get<const MeshStore>().GetHiddenElements(id, mode);
        f.Selection(mode).ForEach([&](uint32_t h) { expect(!hidden.Contains(h)); });
        f.Do(action::mesh::Hide{});
        f.Checkpoint();
        f.Click(f.ViewFraction(vec3{}));
        f.Checkpoint();
        expect(f.Selection(mode).Count() == 0u);
        f.Do(action::mesh::Reveal{});
        f.Checkpoint();
        expect(f.Selection(mode).Count() == 2u);
    }
}

void TestVisibilityLifecycle() {
    Fixture f{"visibility-lifecycle"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}};
    source.Data.AddFace(std::array{0u, 1u, 4u, 3u});
    source.Data.AddFace(std::array{1u, 2u, 5u, 4u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Face);
    const std::array selected{0u};
    f.SelectElements(entity, selected, Element::Face);
    f.P->History.Commit("Select left face", {});
    f.Do(action::mesh::Hide{});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Subdivide{1u});
    f.Checkpoint();
    const auto check = [&](const MeshStore &meshes, uint32_t mesh_id) {
        const Mesh mesh{meshes, mesh_id};
        const auto hidden = meshes.GetHiddenElements(mesh_id, Element::Face);
        expect(mesh.FaceCount() == 5u);
        expect(hidden.Count() == 1u);
        for (const auto face : mesh.faces()) {
            float x = 0.f;
            for (const auto v : mesh.fv_range(face)) x += mesh.GetPosition(v).x;
            expect(hidden.Contains(*face) == (x / mesh.GetValence(face) < 1.f));
        }
        expect(meshes.GetHiddenElements(mesh_id, Element::Vertex).Count() == 2u);
        expect(meshes.GetHiddenElements(mesh_id, Element::Edge).Count() == 3u);
        ExpectSelectionIndex(meshes, mesh_id);
    };
    check(f.R.Context.get<const MeshStore>(), id);
    f.P->Undo();
    f.Checkpoint();
    expect(f.ActiveMesh().FaceCount() == 2u);
    f.P->Redo();
    f.Checkpoint();
    check(f.R.Context.get<const MeshStore>(), id);
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::Duplicate{});
    f.Checkpoint();
    for (const auto [e, handle] : f.R.view<const MeshHandle>().each()) check(f.R.Context.get<const MeshStore>(), handle.StoreId);
    expect(f.P->Save());
    expect(f.P->Close());
    Engine reopened{false};
    expect(reopened.P->Open(f.Dir));
    for (const auto [e, handle] : reopened.R.view<const MeshHandle>().each()) check(reopened.R.Context.get<const MeshStore>(), handle.StoreId);
    expect(reopened.R.Context.get<action::Errors>().Messages.empty());
}

void TestSnapSymmetry() {
    Fixture f{"snap-symmetry"};
    MeshSource source;
    source.Data.Positions = {{-1, 0, 0}, {1.08f, .04f, .02f}, {-2, 1, .2f}, {2.02f, 1.05f, .25f}, {.02f, 2, .3f}, {4, 2, 3}, {.1f, 4, .6f}, {-1.02f, 0, 0}};
    source.Data.AddFace(std::array{0u, 1u, 4u});
    source.Data.AddFace(std::array{2u, 3u, 5u});
    const auto [entity, instance] = f.AddEditable(CreateMesh(f.R, std::move(source)).StoreId, Element::Vertex);
    const std::array selected{0u, 2u, 4u, 5u, 6u, 7u};
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select symmetry vertices", {});
    const auto original = Positions(f.ActiveMesh());
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::SnapSymmetry{.Threshold = 0.f});
    expect(*f.P->History.Present == history);
    // Blender 5.2.2: interpolate toward positive partners, including unselected
    // partners; center eligible vertices and leave unmatched vertices untouched.
    // The first canonical pair owns its partner, leaving the competing vertex 7 unchanged.
    const std::array reference{vec3{-1.06f, .03f, .015f}, vec3{1.06f, .03f, .015f}, vec3{-2.015f, 1.0375f, .2375f}, vec3{2.015f, 1.0375f, .2375f}, vec3{0, 2, .3f}, vec3{4, 2, 3}, vec3{.1f, 4, .6f}};
    auto expected = original;
    for (uint32_t i = 0u; i < reference.size(); ++i) expected[i].second = reference[i];
    f.Stage(action::mesh::SnapSymmetry{.Threshold = .2f, .Factor = .25f, .Negative = false});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected);
    expect(f.Selection(Element::Vertex).Count() == selected.size());
    f.Cancel();
    // Factor zero still snaps to the chosen side; disabling Center preserves x=.02.
    expected = original;
    expected[1].second = {1, 0, 0};
    expected[3].second = {2, 1, .2f};
    f.Do(action::mesh::SnapSymmetry{.Threshold = .2f, .Factor = 0.f, .Negative = true, .Center = false});
    f.Checkpoint();
    ExpectPositions(f.ActiveMesh(), expected);
}

void TestUnsubdivide() {
    struct Reference {
        uint32_t Size, Iterations;
        std::vector<std::vector<uint32_t>> Faces;
    };
    const std::array references{
        Reference{3u, 1u, {{9, 4, 1, 6}, {0, 1, 4}, {14, 9, 6, 11}, {9, 12, 4}, {9, 14, 12}, {1, 3, 6}, {11, 15, 14}, {3, 11, 6}}},
        Reference{3u, 2u, {{9, 14, 4}, {0, 1, 4}, {14, 9, 11}, {9, 4, 1}, {11, 9, 1, 3}, {11, 15, 14}}},
    };
    const auto canonical = [](auto faces) {
        for (auto &face : faces) std::rotate(face.begin(), std::ranges::min_element(face), face.end());
        std::ranges::sort(faces);
        return faces;
    };
    // bmesh.ops.unsubdivide, Blender 5.2.2 d13f752e3b9c. Compare oriented loops.
    for (const auto &reference : references) {
        Fixture f{"unsubdivide"};
        const auto size = reference.Size, width = size + 1u;
        MeshSource source;
        auto &uvs = source.Attrs.TexCoords0.emplace();
        auto &colors = source.Attrs.Colors0.emplace();
        source.Attrs.Colors0ComponentCount = 4u;
        for (uint32_t y = 0u; y < width; ++y)
            for (uint32_t x = 0u; x < width; ++x) {
                source.Data.Positions.push_back({float(x), float(y), 0.f});
                uvs.push_back({float(x), float(y)});
                colors.push_back({float(x) / 4.f, float(y) / 4.f, 0.f, 1.f});
            }
        for (uint32_t y = 0u; y < size; ++y)
            for (uint32_t x = 0u; x < size; ++x) {
                const auto v = y * width + x;
                source.Data.AddFace(std::array{v, v + 1u, v + width + 1u, v + width});
            }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        std::vector<uint32_t> selected;
        for (const auto v : f.ActiveMesh().vertices()) selected.push_back(*v);
        f.SelectElements(entity, selected, Element::Vertex);
        f.P->History.Commit("Select unsubdivide grid", {});
        const auto original = Positions(f.ActiveMesh());
        const auto check = [&] {
            const auto mesh = f.ActiveMesh();
            const auto &a = f.R.Context.get<const MeshStore>().Arenas();
            std::vector<std::vector<uint32_t>> actual;
            for (const auto face : mesh.faces()) {
                std::vector<uint32_t> loop;
                for (const auto h : mesh.fh_range(face)) {
                    const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                    loop.push_back(uint32_t(p.x + float(width) * p.y));
                    expect(a.CornerUvs[0].Get(*h) == vec2{p.x, p.y});
                    expect(a.CornerColors.Get(*h) == vec4{p.x / 4.f, p.y / 4.f, 0.f, 1.f});
                }
                actual.push_back(std::move(loop));
            }
            expect(canonical(actual) == canonical(reference.Faces)) << "Blender unsubdivide grid" << size << reference.Iterations;
            expect(f.Selection(Element::Vertex).Count() == mesh.VertexCount());
            for (const auto v : mesh.vertices()) expect(mesh.GetPosition(v) == original[*v].second);
        };
        const action::mesh::Unsubdivide action{reference.Iterations};
        f.Do(action);
        f.Checkpoint();
        check();
    }
}

void TestUnsubdivideLinesAndBoundaries() {
    for (uint32_t kind = 0u; kind < 4u; ++kind) {
        Fixture f{"unsubdivide-boundaries"};
        MeshSource source;
        if (kind == 0u) {
            for (uint32_t i = 0u; i < 9u; ++i) source.Data.Positions.push_back({float(i), 0.f, 0.f});
            for (uint32_t i = 0u; i < 8u; ++i) source.Data.Edges.push_back({i, i + 1u});
        } else {
            source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 1, 0}, {0, 2, 0}, {1, 2, 0}, {2, 2, 0}};
            source.Data.AddFace(std::array{0u, 1u, 4u, 3u});
            source.Data.AddFace(std::array{1u, 2u, 5u, 4u});
            source.Data.AddFace(std::array{3u, 4u, 7u, 6u});
            source.Data.AddFace(std::array{4u, 5u, 8u, 7u});
            if (kind == 3u) source.Data.AddFace(std::array{1u, 4u, 7u});
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        std::vector<uint32_t> selected;
        for (const auto v : f.ActiveMesh().vertices())
            if (kind != 2u || *v != 4u) selected.push_back(*v);
        f.SelectElements(entity, selected, Element::Vertex);
        f.P->History.Commit("Select unsubdivide boundary", {});
        const auto original = Positions(f.ActiveMesh());
        const auto history = *f.P->History.Present;
        f.Do(action::mesh::Unsubdivide{2u});
        f.Checkpoint();
        if (kind == 0u) {
            ExpectCounts(f.ActiveMesh(), {4u, 3u, 0u, 0u});
            std::vector<float> x;
            for (const auto v : f.ActiveMesh().vertices()) x.push_back(f.ActiveMesh().GetPosition(v).x);
            expect(x == std::vector<float>{0, 1, 5, 8});
            expect(f.Selection(Element::Vertex).Count() == 4u);
        } else if (kind == 1u) expect(f.ActiveMesh().VertexCount() == 8u);
        else {
            expect(*f.P->History.Present == history);
            ExpectPositions(f.ActiveMesh(), original);
        }
    }
}

void TestBeautifyFaces() {
    using Method = action::mesh::BeautifyMethod;
    for (const auto method : {Method::Area, Method::Angle}) {
        const bool warped = method == Method::Angle;
        Fixture f{"beautify-faces"};
        MeshSource source;
        source.Data.Positions = warped ? std::vector<vec3>{{0, 0, 0}, {3, 0, .3f}, {4, 1, -.7f}, {3, 3, .2f}, {1, 4, -.4f}, {-1, 2, .6f}} :
                                         std::vector<vec3>{{-5, 0, 0}, {-4, -3, 0}, {-1, -4, 0}, {3, -3, 0}, {6, 0, 0}, {5, 4, 0}, {2, 6, 0}, {-2, 5, 0}};
        const uint32_t n = uint32_t(source.Data.Positions.size());
        for (uint32_t i = 1u; i + 1u < n; ++i) source.Data.AddFace(std::array{0u, i, i + 1u});
        // Unselected neighboring triangle and polygon retain their own loops.
        source.Data.Positions.insert(source.Data.Positions.end(), {{-7, 0, 0}, {-7, -3, 0}});
        source.Data.AddFace(std::array{1u, 0u, n});
        source.Data.AddFace(std::array{0u, n - 1u, n, n + 1u});
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Face);
        std::vector<uint32_t> selected;
        for (uint32_t i = 0u; i < n - 2u; ++i) selected.push_back(i);
        f.SelectElements(entity, selected, Element::Face);
        f.P->History.Commit("Select beautify faces", {});
        const auto original = Positions(f.ActiveMesh());
        const auto counts = CountsOf(f.ActiveMesh());
        const auto untouched_triangle = FacePositions(f.ActiveMesh(), f.ActiveMesh().FaceAt(n - 2u));
        const auto untouched_polygon = FacePositions(f.ActiveMesh(), f.ActiveMesh().FaceAt(n - 1u));
        std::vector<std::vector<uint32_t>> expected;
        // bmesh.ops.beautify_fill references, Blender 5.2.2 d13f752e3b9c.
        if (warped) expected = {{0, 1, 3}, {0, 3, 4}, {0, 4, 5}, {1, 2, 3}};
        else expected = {{0, 1, 2}, {0, 2, 7}, {2, 3, 7}, {3, 4, 5}, {3, 5, 6}, {3, 6, 7}};
        const action::mesh::BeautifyFaces action{method};
        f.Do(action);
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        ExpectPositions(mesh, original);
        ExpectCounts(mesh, counts);
        expect(f.Selection(Element::Face).Count() == selected.size());
        expect(FacePositions(mesh, mesh.FaceAt(n - 2u)) == untouched_triangle);
        expect(FacePositions(mesh, mesh.FaceAt(n - 1u)) == untouched_polygon);
        std::vector<std::vector<uint32_t>> actual;
        for (uint32_t i = 0u; i < n - 2u; ++i) {
            std::vector<uint32_t> face;
            for (const auto v : mesh.fv_range(mesh.FaceAt(i))) face.push_back(uint32_t(std::ranges::find(original, v, &VertexPositions::value_type::first) - original.begin()));
            std::ranges::sort(face);
            actual.push_back(face);
        }
        std::ranges::sort(actual);
        expect(actual == expected) << "Blender beautify partition";
        const auto edited = *f.P->History.Present;
        f.Do(action);
        f.Checkpoint();
        expect(*f.P->History.Present == edited) << "beautify converges";
    }
}

void TestBeautifyBoundaries() {
    for (const uint32_t kind : {0u, 1u, 2u}) {
        Fixture f{"beautify-boundaries"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {3, 0, 0}, {2, 1, 0}, {0, 2, 0}, {1, 1, 1}};
        source.Data.AddFace(std::array{0u, 1u, 3u});
        source.Data.AddFace(std::array{1u, 2u, 3u});
        if (kind == 1u) source.Data.AddFace(std::array{3u, 1u, 4u}); // Three users: not a manifold diagonal.
        if (kind == 2u) source.Data.AddFace(std::array{0u, 2u, 4u}); // Proposed diagonal already exists outside the selection.
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Face);
        const std::vector<uint32_t> selected = kind ? std::vector{0u, 1u} : std::vector{0u};
        f.SelectElements(entity, selected, Element::Face);
        f.P->History.Commit("Select bounded beautify", {});
        const auto before = *f.P->History.Present;
        f.Do(action::mesh::BeautifyFaces{});
        f.Checkpoint();
        expect(*f.P->History.Present == before);
    }
}

void TestBeautifySeams() {
    for (const bool reversed : {false, true}) {
        Fixture f{"beautify-seams"};
        MeshSource source{.Weld = true};
        const std::array points{vec3{0, 0, 0}, vec3{3, 0, 0}, vec3{2, 1, 0}, vec3{0, 2, 0}};
        const std::array loops{std::array{0u, 1u, 3u}, reversed ? std::array{3u, 2u, 1u} : std::array{1u, 2u, 3u}};
        auto &uvs = source.Attrs.TexCoords0.emplace();
        for (uint32_t face = 0u; face < 2u; ++face) {
            for (const auto v : loops[face]) {
                source.Data.Positions.push_back(points[v]);
                uvs.push_back({float(face * 10u + v), float(face)});
            }
            source.Data.AddFace(std::array{face * 3u, face * 3u + 1u, face * 3u + 2u});
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Face);
        const std::array selected{0u, 1u};
        f.SelectElements(entity, selected, Element::Face);
        f.P->History.Commit("Select seam faces", {});
        expect(f.ActiveMesh().VertexCount() == 4u);
        f.Do(action::mesh::BeautifyFaces{});
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        const auto &a = f.R.Context.get<const MeshStore>().Arenas();
        uint32_t negative = 0u;
        // Blender's boundary loop provenance: vertex 1 comes from the second
        // face and vertex 3 from the first, despite their original UV seam.
        for (const auto face : mesh.faces()) {
            negative += mesh.GetNormal(face).z < 0.f;
            for (const auto h : mesh.fh_range(face)) {
                const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                const auto v = uint32_t(std::ranges::find(points, p) - points.begin());
                const uint32_t from = v == 1u || v == 2u ? 1u : 0u;
                expect(a.CornerUvs[0].Get(*h) == vec2{float(from * 10u + v), float(from)});
            }
        }
        expect(negative == uint32_t(reversed)) << "preserves relative face winding";
    }
}

void TestFaceAttributes() {
    Fixture f{"face-attributes"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {2, 0, 0}, {2, 1, 0}, {0, 2, 0}};
    const std::array<std::vector<uint32_t>, 3> faces{{{0, 1, 2, 3}, {1, 4, 5, 2}, {3, 2, 6}}};
    for (const auto &face : faces) source.Data.AddFace(face);
    std::vector<vec2> uvs;
    std::vector<vec4> colors;
    for (uint32_t i = 0u; i < source.Data.Positions.size(); ++i) {
        uvs.push_back({float(i), float(i * 3u)});
        colors.push_back({float(i) / 8.f, .5f, .25f, 1});
    }
    source.Attrs.TexCoords0 = uvs;
    source.Attrs.TexCoords3 = uvs;
    source.Attrs.Colors0 = colors;
    source.Attrs.Colors0ComponentCount = 4u;
    source.Attrs.Tangents = std::vector<vec4>(uvs.size(), vec4{1, 0, 0, 1});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Face);
    const std::array selected{0u, 2u};
    f.SelectElements(entity, selected, Element::Face);
    f.P->History.Commit("Select attribute faces", {});
    const auto original = Positions(f.ActiveMesh());
    const std::array<std::array<uint32_t, 4>, 3> quad{{{3, 0, 1, 2}, {1, 2, 3, 0}, {3, 2, 1, 0}}};
    const std::array<std::array<uint32_t, 3>, 3> triangle{{{2, 0, 1}, {1, 2, 0}, {2, 1, 0}}};
    for (const bool edit_colors : {false, true})
        for (const uint32_t operation : {edit_colors ? 0u : 1u, 2u}) {
            if (edit_colors) {
                if (operation == 2u) f.Stage(action::mesh::ReverseColors{});
                else f.Stage(action::mesh::RotateColors{});
            } else if (operation == 2u) f.Stage(action::mesh::ReverseUVs{3u});
            else f.Stage(action::mesh::RotateUVs{3u, true});
            f.Checkpoint();
            const auto mesh = f.ActiveMesh();
            const auto &a = f.R.Context.get<const MeshStore>().Arenas();
            ExpectPositions(mesh, original);
            for (uint32_t face = 0u; face < faces.size(); ++face) {
                uint32_t i = 0u;
                for (const auto h : mesh.fh_range(mesh.FaceAt(face))) {
                    const auto before = faces[face][i], after = faces[face][face == 1u ? i : face == 0u ? quad[operation][i] :
                                                                                                          triangle[operation][i]];
                    expect(a.CornerUvs[0].Get(*h) == uvs[before]);
                    expect(a.CornerUvs[3].Get(*h) == uvs[edit_colors ? before : after]);
                    expect(a.CornerColors.Get(*h) == colors[edit_colors ? after : before]);
                    expect(a.CornerTangents.Get(*h) == (!edit_colors && face != 1u ? vec4{} : vec4{1, 0, 0, 1}));
                    ++i;
                }
            }
            f.Cancel();
        }
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::ReverseUVs{4u});
    expect(*f.P->History.Present == history);
}

void TestDecimate() {
    for (const uint32_t kind : {0u, 2u, 3u, 4u, 5u}) {
        Fixture f{"decimate"};
        MeshSource source;
        constexpr uint32_t width = 7u;
        source.Attrs.TexCoords0.emplace();
        source.Attrs.Colors0.emplace();
        source.Attrs.Colors0ComponentCount = 4u;
        for (uint32_t y = 0u; y < width; ++y)
            for (uint32_t x = 0u; x < width; ++x) {
                const float z = kind == 3u ? .12f * float(x * x + y * y) : 0.f;
                source.Data.Positions.push_back({float(x), float(y), z});
                source.Attrs.TexCoords0->push_back({float(x) / 6.f, float(y) / 6.f});
                source.Attrs.Colors0->push_back({float(x) / 6.f, float(y) / 6.f, 0.f, 1.f});
            }
        for (uint32_t y = 0u; y + 1u < width; ++y)
            for (uint32_t x = 0u; x + 1u < width; ++x) {
                const uint32_t v = y * width + x;
                if (kind == 2u) source.Data.AddFace(std::array{v, v + 1u, v + width + 1u, v + width});
                else {
                    source.Data.AddFace(std::array{v, v + 1u, v + width + 1u});
                    source.Data.AddFace(std::array{v, v + width + 1u, v + width});
                }
                if (kind == 4u) {
                    source.Primitives.ElementPrimitiveIndices.push_back(x < 3u ? 0u : 1u);
                    source.Primitives.ElementPrimitiveIndices.push_back(x < 3u ? 0u : 1u);
                }
            }
        if (kind == 4u) source.Primitives.MaterialIndices = {0u, 0u};
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
        if (kind == 5u) {
            f.EnterEdit(Element::Face);
            const std::array selected_face{10u};
            f.SelectElements(entity, selected_face, Element::Face);
            f.Do(action::mesh::Hide{});
            f.EnterEdit(Element::Vertex);
        }
        std::vector<uint32_t> selected;
        for (uint32_t y = 0u; y < width; ++y)
            for (uint32_t x = 0u; x < width; ++x)
                if (kind != 3u || (x >= 2u && x <= 4u && y >= 2u && y <= 4u)) selected.push_back(y * width + x);
        ApplyEditSelectionLists(f.R, std::array{std::pair{entity, std::span<const uint32_t>{selected}}}, Element::Vertex);
        f.P->Settle();
        f.P->History.Commit("Select decimation region", {});
        const auto original = Positions(f.ActiveMesh());
        const auto counts = CountsOf(f.ActiveMesh());
        const auto unchanged = *f.P->History.Present;
        f.Do(action::mesh::Decimate{1.f});
        expect(*f.P->History.Present == unchanged);
        const auto check = [&](const state::Scene &r) {
            const auto &meshes = r.Context.get<const MeshStore>();
            const auto &a = meshes.Arenas();
            const Mesh mesh{meshes, id};
            expect(mesh.VertexCount() < counts.Vertices) << "decimate fixture" << kind;
            expect(mesh.TriangleIndexCount() / 3u < counts.Triangles);
            // Blender 5.2.2 d13f752e3b9c: these half-ratio grids end at
            // 35, 36 and 36 triangles. Collapse order and retained polygons differ.
            if (kind < 3u) {
                expect(mesh.TriangleIndexCount() / 3u <= 36u);
                expect(mesh.TriangleIndexCount() / 3u >= 35u);
            }
            expect(int(mesh.VertexCount()) - int(mesh.EdgeCount()) + int(mesh.FaceCount()) == 1);
            std::map<uint32_t, vec2> uvs;
            for (const auto face : mesh.faces()) {
                for (const auto h : mesh.fh_range(face)) {
                    const auto v = mesh.GetToVertex(h);
                    const auto p = mesh.GetPosition(v);
                    expect(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
                    if (kind == 3u) expect(std::abs(p.z - .12f * (p.x * p.x + p.y * p.y)) < .3f);
                    const auto uv = a.CornerUvs[0].Get(*h);
                    const auto [it, inserted] = uvs.try_emplace(*v, uv);
                    expect(inserted || Length(it->second - uv) < 1e-6f) << "decimation introduced an attribute seam" << kind << uv.x << uv.y << it->second.x << it->second.y;
                    const auto color = a.CornerColors.Get(*h);
                    expect(color == vec4{uv.x, uv.y, 0.f, 1.f});
                }
                expect(mesh.GetNormal(face).z > 0.f);
            }
            for (const auto &[vertex, position] : original) {
                const auto x = uint32_t(position.x), y = uint32_t(position.y);
                const bool fixed = (kind == 3u && (x < 2u || x > 4u || y < 2u || y > 4u)) || (kind == 4u && x == 3u);
                if (fixed) {
                    expect(meshes.IsLiveElement(id, Element::Vertex, *vertex));
                    expect(mesh.GetPosition(vertex) == position);
                }
            }
            expect(meshes.GetHiddenElements(id, Element::Face).Count() == (kind == 5u ? 1u : 0u));
        };
        const action::mesh::Decimate action{.5f};
        f.Do(action);
        f.Checkpoint();
        check(f.R);
    }
    for (const bool nonmanifold : {false, true}) {
        Fixture f{"decimate-invalid"};
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, -1, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u});
        source.Data.AddFace(std::array{1u, 0u, 3u});
        if (nonmanifold) source.Data.AddFace(std::array{0u, 1u, 4u});
        else {
            source.Data.AddFace(std::array{1u, 3u, 2u});
            source.Data.AddFace(std::array{0u, 2u, 3u});
        }
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        f.AddEditable(id, Element::Vertex);
        f.Do(action::selection::SelectAll{});
        const auto before = *f.P->History.Present;
        const auto counts = CountsOf(f.ActiveMesh());
        f.Do(action::mesh::Decimate{0.f});
        f.Checkpoint();
        ExpectCounts(f.ActiveMesh(), counts);
        expect(*f.P->History.Present == before);
    }
}

void TestSeparate() {
    using Mode = action::mesh::SeparateMode;
    struct Case {
        Mode Operation;
        Element Domain;
        uint32_t Vertices;
        std::vector<std::array<uint32_t, 3>> Faces;
        std::vector<std::array<uint32_t, 2>> Edges;
        std::vector<uint32_t> Selected;
        std::vector<std::vector<uint32_t>> Expected;
        uint32_t OutputEdges;
    };
    const std::array cases{
        // Extract both a face with an attached wire and an independent wire from a mixed source.
        Case{Mode::LooseParts, Element::Vertex, 10u, {{0, 1, 2}, {3, 4, 5}}, {{5, 6}, {7, 8}}, {}, {{0, 1, 2}, {3, 4, 5, 6}, {7, 8}, {9}}, 8u},
        Case{Mode::Material, Element::Face, 7u, {{0, 1, 2}, {2, 1, 3}, {4, 5, 6}}, {}, {}, {{0, 1, 2, 4, 5, 6}, {1, 2, 3}}, 9u},
        Case{Mode::Selected, Element::Vertex, 6u, {}, {}, {1, 4}, {{0, 2, 3, 5}, {1, 4}}, 0u},
        Case{Mode::Selected, Element::Vertex, 6u, {}, {{0, 1}, {1, 2}, {3, 4}, {4, 5}}, {1, 2}, {{0, 1, 3, 4, 5}, {1, 2}}, 4u},
        Case{Mode::Selected, Element::Face, 4u, {{0, 1, 2}, {2, 1, 3}}, {}, {1}, {{0, 1, 2}, {1, 2, 3}}, 6u},
    };
    for (const auto &test : cases) {
        Fixture f{"separate"};
        MeshSource source;
        source.Data.Edges = test.Edges;
        for (const auto face : test.Faces) source.Data.AddFace(face);
        auto &uvs = source.Attrs.TexCoords0.emplace();
        auto &colors = source.Attrs.Colors0.emplace();
        source.Attrs.Colors0ComponentCount = 4u;
        for (uint32_t i = 0u; i < test.Vertices; ++i) {
            const vec3 p{float(i), float(i % 3u == 1u), 0.f};
            source.Data.Positions.push_back(p);
            uvs.push_back({p.x, p.y});
            colors.push_back({p.x / 10.f, p.y, 0.f, 1.f});
        }
        const auto material = f.R.Context.get<GpuBuffers>().Materials.Append(PBRMaterial{});
        source.Primitives.MaterialIndices = {material, 0u, material};
        source.Primitives.ElementPrimitiveIndices.assign(test.Faces.empty() ? test.Vertices : test.Faces.size(), 0u);
        if (test.Faces.size() >= 2u) source.Primitives.ElementPrimitiveIndices[1] = 1u;
        if (test.Faces.size() == 3u) source.Primitives.ElementPrimitiveIndices[2] = 2u;
        const auto [entity, instance] = f.AddEditable(CreateMesh(f.R, std::move(source)).StoreId, test.Domain);
        if (test.Operation == Mode::LooseParts) {
            f.EnterEdit(Element::Face);
            f.SelectElements(entity, std::array{0u}, Element::Face);
            f.Do(action::mesh::Hide{});
            f.EnterEdit(test.Domain);
        }
        f.SelectElements(entity, test.Selected, test.Domain);
        f.P->History.Commit("Prepare separation", {});
        const auto original = Positions(f.ActiveMesh());
        const auto check = [&](const state::Scene &r) {
            const auto &meshes = r.Context.get<const MeshStore>();
            const auto &a = meshes.Arenas();
            std::vector<std::vector<uint32_t>> actual;
            uint32_t edges = 0u, faces = 0u, hidden = 0u;
            for (const auto [e, handle] : r.view<const MeshHandle>().each()) {
                const Mesh mesh{meshes, handle.StoreId};
                auto &vertices = actual.emplace_back();
                for (const auto v : mesh.vertices()) vertices.push_back(uint32_t(mesh.GetPosition(v).x));
                std::ranges::sort(vertices);
                const auto &record = meshes.Get(handle.StoreId);
                const auto palette = a.PrimitiveMaterials.Get(record.PrimitiveMaterials);
                for (const auto face : mesh.faces()) {
                    const bool second = std::ranges::any_of(mesh.fv_range(face), [&](auto v) { return mesh.GetPosition(v).x == 3.f; });
                    expect(palette[a.FacePrimitives.Get(*face)] == (second ? 0u : material));
                    for (const auto h : mesh.fh_range(face)) {
                        const auto p = mesh.GetPosition(mesh.GetToVertex(h));
                        expect(a.CornerUvs[0].Get(*h) == vec2{p.x, p.y});
                        expect(a.CornerColors.Get(*h) == vec4{p.x / 10.f, p.y, 0.f, 1.f});
                    }
                }
                if (test.Faces.empty())
                    for (const auto v : mesh.vertices()) {
                        const auto p = mesh.GetPosition(v);
                        expect(palette[a.VertexPrimitives.Get(*v)] == material);
                        expect(a.VertexColors.Get(*v) == vec4{p.x / 10.f, p.y, 0.f, 1.f});
                    }
                edges += mesh.EdgeCount();
                faces += mesh.FaceCount();
                hidden += meshes.GetHiddenElements(handle.StoreId, Element::Face).Count();
                ExpectRenderedGeometry(meshes, mesh);
                if (test.Operation == Mode::Material) ExpectSelectionIndex(meshes, handle.StoreId);
            }
            std::ranges::sort(actual);
            expect(actual == test.Expected) << "separate mode/domain" << uint32_t(test.Operation) << uint32_t(test.Domain);
            expect(edges == test.OutputEdges && faces == test.Faces.size()) << "separate mode/domain" << uint32_t(test.Operation) << uint32_t(test.Domain) << "edges/faces" << edges << faces;
            expect(hidden == uint32_t(test.Operation == Mode::LooseParts));
        };
        f.Do(action::mesh::Separate{test.Operation});
        f.Checkpoint();
        check(f.R);
        if (test.Operation == Mode::Material) {
            f.CheckUndoRedo([&](const state::Scene &r, bool separated) {
                if (separated) check(r);
                else {
                    expect(r.view<const MeshHandle>().size() == 1u);
                    ExpectPositions(f.ActiveMesh(), original);
                }
            });
            f.CheckSaved([&](const state::Scene &r, bool) { check(r); });
        }
    }
    Fixture f{"separate-noop"};
    f.Cube(Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto before = *f.P->History.Present;
    f.Do(action::mesh::Separate{Mode::LooseParts});
    f.Do(action::mesh::Separate{Mode::Material});
    f.Checkpoint();
    expect(*f.P->History.Present == before);
}

vec2 ReferenceUV(uint32_t i) { return {float(i), .25f}; }
vec4 ReferenceColor(uint32_t i) { return {float(i) * .001f, .2f, .4f, 1.f}; }

void AddReferenceAttributes(MeshSource &source) {
    source.Attrs.TexCoords0.emplace();
    source.Attrs.Colors0.emplace();
    for (uint32_t i = 0u; i < source.Data.Positions.size(); ++i) {
        source.Attrs.TexCoords0->push_back(ReferenceUV(i));
        source.Attrs.Colors0->push_back(ReferenceColor(i));
    }
}

void ExpectReferenceAttributes(const MeshStore &store, const Mesh &mesh) {
    for (const auto face : mesh.faces())
        for (const auto h : mesh.fh_range(face)) {
            const auto v = mesh.VertexOrdinal(mesh.GetToVertex(h));
            expect(store.Arenas().CornerUvs[0].Get(*h) == ReferenceUV(v));
            expect(store.Arenas().CornerColors.Get(*h) == ReferenceColor(v));
        }
}

// Each numerical case also leaves an unselected point untouched.
void CheckFit(const char *name, MeshSource source, const auto &action, std::vector<vec3> expected, std::span<const uint32_t> controls = {}) {
    Fixture f{name};
    source.Weld = false;
    source.Data.Positions.push_back({20, 20, 20});
    expected.push_back({20, 20, 20});
    AddReferenceAttributes(source);
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto mode = controls.empty() ? Element::Edge : Element::Vertex;
    const auto [entity, instance] = f.AddEditable(id, mode);
    if (controls.empty()) f.Do(action::selection::SelectAll{});
    else f.SelectElements(entity, controls, mode);
    const auto counts = CountsOf(f.ActiveMesh());
    f.Do(action);
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, counts);
    for (uint32_t i = 0u; i < expected.size(); ++i)
        expect(Near(mesh.GetPosition(mesh.VertexAt(i)), expected[i])) << name << "vertex" << i;
    ExpectReferenceAttributes(f.R.Context.get<const MeshStore>(), mesh);
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), mesh);
}

void TestCircularize() {
    MeshSource polygon;
    polygon.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {3, 1, 0}, {2, 3, 0}, {-1, 2, 0}};
    polygon.Data.AddFace(std::array{0u, 1u, 2u, 3u, 4u});
    // Independent least-squares reference, also checked against Blender.
    CheckFit("circularize", std::move(polygon), action::mesh::Circularize{}, {{-.4543723752f, .2665205426f, 0}, {1.74700349f, -.2120457217f, 0}, {2.882409607f, 1.733702031f, 0}, {1.382753314f, 3.41480654f, 0}, {-.6794913643f, 2.508038512f, 0}});
    MeshSource wire;
    wire.Data.Positions = {{2, 0, 0}, {1.2f, 1, 0}, {0, 2, 0}, {-1, 1.1f, 0}, {-2, 0, 0}};
    wire.Data.Edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
    CheckFit("circularize-open", std::move(wire), action::mesh::Circularize{.Method = action::mesh::CircleFit::Contract, .Regular = false}, {{1.08173014f, .2805637001f, 0}, {1.074599406f, .9587836917f, 0}, {.001674496185f, 1.730462079f, 0}, {-1, 1.1f, 0}, {-1.065226088f, .2831478019f, 0}});
}

void TestCurveBetweenSelected() {
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {.5f, 1, 0}, {1, 0, .2f}, {2, 1, .5f}, {3, -1, 0}, {4, 0, 0}, {5, 1, 0}};
    for (uint32_t i = 1u; i < 7u; ++i) source.Data.Edges.push_back({i - 1u, i});
    // SciPy natural-cubic reference: trim outside the selected controls.
    CheckFit("curve", std::move(source), action::mesh::CurveBetweenSelected{}, {{0, 0, 0}, {.5f, 1, 0}, {1.203125f, 1.09375f, .34375f}, {2, 1, .5f}, {2.953125f, .59375f, .34375f}, {4, 0, 0}, {5, 1, 0}}, std::array{1u, 3u, 5u});
}

void TestFlatten() {
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {2, 0, 0}, {2, 2, 1}, {0, 2, 0}, {10, 0, 0}, {10, 2, 0}, {11, 2, 2}, {10, 0, 2}};
    source.Data.Edges = {{0, 1}, {1, 2}, {2, 3}, {4, 5}, {5, 6}, {6, 7}};
    // Independent PCA fits: disconnected components must use separate planes.
    CheckFit("flatten", std::move(source), action::mesh::Flatten{}, {{.06480586011f, .06480586011f, -.2449150189f}, {1.941974115f, -.05802588532f, .2192920636f}, {2.051245911f, 2.051245911f, .8063308916f}, {-.05802588532f, 1.941974115f, .2192920636f}, {9.755084981f, .06480586011f, .06480586011f}, {10.21929206f, 1.941974115f, -.05802588532f}, {10.80633089f, 2.051245911f, 2.051245911f}, {10.21929206f, -.05802588532f, 1.941974115f}});
}

void TestRelaxEdgeLoops() {
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {3, 0, 0}, {3, 2, 0}, {0, 2, 0}};
    source.Data.Edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}};
    // Independent periodic spline solve for alternating relaxation phases.
    CheckFit("relax", std::move(source), action::mesh::RelaxEdgeLoops{.EvenSpacing = false}, {{.257891029f, .292481363f, 0}, {.982795715f, -.248387098f, 0}, {2.63177109f, .308072478f, 0}, {2.78709674f, 1.73870969f, 0}, {.611708343f, 1.09812188f, 0}});
}

void TestSpaceEvenly() {
    // Blender references cover both interpolation methods and fixed open endpoints.
    const std::array<std::vector<vec3>, 2> expected{{{{0, 0, 0}, {1.11557984f, .50538290f, 0}, {2.45127296f, .90254611f, 0}, {3.56619334f, 1.43380666f, 0}, {5, 0, 0}}, {{0, 0, 0}, {1.05305076f, .56604123f, 0}, {2.54712796f, .72494555f, 0}, {3.43722606f, 2.06138992f, 0}, {5, 0, 0}}}};
    for (uint32_t cubic = 0u; cubic < 2u; ++cubic) {
        MeshSource source;
        source.Data.Positions = {{0, 0, 0}, {.25f, 1, 0}, {2, 0, 0}, {3, 2, 0}, {5, 0, 0}};
        source.Data.Edges = {{0, 1}, {1, 2}, {2, 3}, {3, 4}};
        CheckFit("space-evenly", std::move(source), action::mesh::SpaceEvenly{.Interpolation = cubic ? action::mesh::EdgeLoopInterpolation::Cubic : action::mesh::EdgeLoopInterpolation::Linear}, expected[cubic]);
    }
}

void TestGridFill() {
    Fixture f{"grid-fill"};
    MeshSource source;
    auto &data = source.Data;
    data.Positions = {{0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {2, 1, 0}, {2, 2, 0}, {1, 2, 0}, {0, 2, 0}, {0, 1, 0}};
    for (uint32_t i = 0u; i < 8u; ++i) data.Positions.push_back(3.f * data.Positions[i] - vec3{2, 2, 0});
    // Half the inner boundary belongs to a surface; the other half is wire.
    for (uint32_t i = 0u; i < 4u; ++i) data.AddFace(std::array{8u + i, 8u + (i + 1u) % 8u, (i + 1u) % 8u, i});
    for (uint32_t i = 4u; i < 8u; ++i) data.Edges.push_back({i, (i + 1u) % 8u});
    data.Positions.insert(data.Positions.end(), {{10, 0, 0}, {11, 0, 0}});
    data.Edges.push_back({16u, 17u});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Edge);
    const auto original = Positions(f.ActiveMesh());
    const auto before = CountsOf(f.ActiveMesh());
    std::vector<std::array<uint32_t, 2>> boundary;
    for (uint32_t i = 0u; i < 8u; ++i) boundary.push_back({i, (i + 1u) % 8u});
    SelectEdgePairs(f, entity, boundary);
    f.Do(action::mesh::GridFill{2u});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {before.Vertices + 1u, before.Edges + 4u, before.Faces + 4u, before.Triangles + 8u});
    for (const auto &[v, p] : original) expect(Near(mesh.GetPosition(v), p));
    expect(Near(mesh.GetPosition(mesh.VertexAt(mesh.VertexCount() - 1u)), {1, 1, 0}));
    ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), mesh);
}

void TestBridgeChains() {
    Fixture f{"bridge-chains"};
    MeshSource source;
    source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}, {10, 0, 0}, {11, 0, 0}, {12, 0, 0}};
    source.Data.AddFace(std::array{3u, 2u, 1u, 0u});
    source.Data.Edges = {{4, 5}, {5, 6}, {6, 7}, {7, 4}, {8, 9}};
    AddReferenceAttributes(source);
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Edge);
    const auto before = Positions(f.ActiveMesh());
    const std::array<std::array<uint32_t, 2>, 8> rails{{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6}, {6, 7}, {7, 4}}};
    SelectEdgePairs(f, entity, rails);
    f.Do(action::mesh::BridgeEdgeLoops{});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {11u, 13u, 5u, 10u});
    ExpectPositions(mesh, before);
    uint32_t joined = 0u;
    for (const auto edge : mesh.edges()) {
        const auto h = mesh.GetHalfedge(edge, 0u);
        joined += bool(mesh.GetOppositeHalfedge(h)) && bool(mesh.GetFace(h));
    }
    expect(joined == 8u); // Four cap edges and four new bridge seams.
    ExpectReferenceAttributes(f.R.Context.get<const MeshStore>(), mesh);
    ExpectRenderedGeometry(f.R.Context.get<const MeshStore>(), mesh);
    ExpectPolygonTessellation(f.R.Context.get<const MeshStore>(), mesh);
}

void TestMixedMeshSources() {
    for (const bool weld : {false, true}) {
        Fixture f{"mixed-source"};
        MeshSource source;
        source.Weld = weld;
        source.Data.Positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 0, 0}, {3, 0, 0}, {0, 0, 0}};
        source.Data.AddFace(std::array{0u, 1u, 2u});
        source.Data.Edges = {{5, 3}, {3, 4}};
        AddReferenceAttributes(source);
        const auto id = CreateMesh(f.R, std::move(source)).StoreId;
        f.AddEditable(id, Element::Vertex);
        f.Checkpoint();
        const auto mesh = f.ActiveMesh();
        const auto &store = f.R.Context.get<const MeshStore>();
        ExpectCounts(mesh, {weld ? 5u : 6u, 5u, 1u, 1u});
        ExpectRenderedGeometry(store, mesh);
        ExpectReferenceAttributes(store, mesh);
        for (const auto edge : mesh.edges()) {
            const auto h = mesh.GetHalfedge(edge, 0u);
            if (mesh.GetFace(h)) continue;
            for (const auto corner : {h, mesh.GetOppositeHalfedge(h)}) {
                const auto p = mesh.GetPosition(mesh.GetToVertex(corner));
                // Welding the wire endpoint onto a surface must preserve its source corner value.
                expect(store.Arenas().CornerUvs[0].Get(*corner).x == (p.x == 0.f ? 5.f : p.x + 1.f));
            }
        }
    }
}

void TestMixedMeshletBuild() {
    Fixture f{"mixed-meshlet-build"};
    auto &meshes = f.R.Context.get<MeshStore>();
    const MeshData data{std::vector<vec3>{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {2, 1, 0}, {2, 2, 0}, {3, 3, 0}}, {{0u, 1u, 2u}}};
    const std::array<std::array<uint32_t, 2>, 2> wires{{{2u, 3u}, {3u, 4u}}};
    std::array<uint32_t, 2> ids;
    for (auto &id : ids) {
        id = meshes.CreateMeshSource(MeshData{data.Positions});
        meshes.AllocateConnectivity(id, 7u, 1u, false, {}, wires);
        const auto &owner = meshes.Get(id);
        const auto &a = meshes.Arenas();
        auto corners = a.FaceCorners.Buffer.GetMutableSpan<uint32_t>(a.FaceCorners.Dense(owner.FaceCorners));
        for (uint32_t c = 0u; c < 3u; ++c) corners[c] = a.Vertices.First(owner.Vertices) + c;
    }
    BuildConnectivityNow(f.R, ids);
    for (const auto id : ids) {
        meshes.CreateMesh(id, data, {}, {}, {}, false);
        meshes.WriteRecord(id).Classification = uint32_t(CornerClassMode::UniformFace);
    }
    const auto expected = [&](uint32_t id) {
        const Mesh mesh{meshes, id};
        std::array<std::vector<uint32_t>, 3> elements{{{meshes.Arenas().Triangles.First(meshes.Get(id).TriangleData)}, {}, {mesh.VertexFirst() + 5u}}};
        for (const auto e : mesh.edges())
            if (!mesh.GetConnectivity().FaceOf(mesh.GetHalfedge(e, 0u))) elements[1].push_back(*e);
        return elements;
    };
    const auto check = [&](uint32_t id) {
        const Mesh mesh{meshes, id};
        ExpectCounts(mesh, {6u, 5u, 1u, 1u});
        expect(RenderedPrimitiveCounts(meshes, id) == std::array{1u, 2u, 1u});
    };
    // Rebuild interleaved mixed owners, clone, release, and rebuild again. Keep this
    // allocation reuse before undo: it caught a missing pre-construction history capture.
    for (uint32_t pass = 0u; pass < 3u; ++pass) {
        mtl::ComputeChain chain{meshes.BufferContext()};
        std::array<std::array<ElementWork, 3>, 2> work;
        for (uint32_t i = 0u; i < ids.size(); ++i) {
            const auto elements = expected(ids[i]);
            for (uint32_t t = 0u; t < 3u; ++t)
                work[i][t] = SeedElementWorkHandles(chain.Scratch, meshes.WithRenderDomain(meshes.Get(ids[i]), t, [](const auto &arena, ElementSetRef) { return arena.Capacity(); }), elements[t]);
        }
        std::vector<MeshletBuildSource> sources;
        for (uint32_t t = 0u; t < 3u; ++t)
            for (uint32_t i = 0u; i < ids.size(); ++i) {
                const auto topology = (t + 2u + i) % 3u;
                sources.push_back({.Destination = &meshes.WriteRecord(ids[i]), .Topology = topology, .ElementCount = topology == 1u ? 2u : 1u, .Elements = work[i][topology]});
            }
        if (pass) std::ranges::reverse(sources);
        if (pass == 2u)
            for (auto &source : sources) {
                source.Elements = {};
                source.ElementCount = meshes.WithRenderDomain(*source.Destination, source.Topology, [](const auto &arena, ElementSetRef set) { return arena.Count(set); });
            }
        BuildGpuMeshlets(f.R, chain, sources);
        chain.Submit();
        for (const auto id : ids) check(id);
    }
    for (const auto id : ids) meshes.PlanClone(Mesh{meshes, id});
    meshes.CommitReserves();
    CloneCopies copies;
    const auto clones = meshes.CloneMeshes(copies, ids);
    mtl::ComputeChain chain{meshes.BufferContext()};
    copies.Encode(chain, GetMeshPipelines(f.R));
    chain.Submit();
    for (const auto id : clones) check(id);
    meshes.ReleaseRender(std::array{&meshes.WriteRecord(ids[0]), &meshes.WriteRecord(ids[1])});
    for (const auto id : ids) expect(meshes.Get(id).RenderTopologies == 0u);
    for (const auto id : clones) check(id);
    // Exercise the scene gather, shading, sparse transforms and restored render state.
    const auto scene_id = clones[0];
    meshes.ReleaseRender(std::array{&meshes.WriteRecord(scene_id)});
    std::ranges::fill(meshes.EditFaceSharpness(scene_id), 0u);
    auto sharpness = meshes.EditEdgeSharpness(scene_id);
    std::ranges::fill(sharpness, 0u);
    sharpness[0] = 1u;
    meshes.UpdateCornerClassification(f.R, chain, std::array{scene_id});
    meshes.EnsureSelectionState(f.R, chain, std::array{scene_id});
    chain.Submit();
    const auto [entity, instance] = AddMesh(f.R, scene_id, MeshInstanceCreateInfo{});
    f.P->Settle();
    f.Checkpoint();
    check(scene_id);
    expect(f.R.Context.get<GpuBuffers>().MeshletTopologyMask == 7u);
    const auto check_geometry = [&](vec3 offset) {
        const Mesh mesh{meshes, scene_id};
        check(scene_id);
        for (uint32_t v = 0u; v < data.Positions.size(); ++v)
            expect(Near(mesh.GetPosition(mesh.VertexAt(v)), data.Positions[v] + offset));
    };
    check_geometry({});
    f.Do(action::selection::Select{instance});
    f.EnterEdit(Element::Vertex);
    f.Do(action::selection::SelectAll{});
    const vec3 shift{10, 2, 3};
    f.Stage(action::view::TransformElements{{.P = shift}});
    f.Commit();
    f.Checkpoint();
    check_geometry(shift);
    f.CheckUndoRedo([&](const state::Scene &, bool moved) { check_geometry(moved ? shift : vec3{}); });

    f.Do(action::view::SetEditMode{.Mode = Element::Face});
    f.Do(action::selection::SelectAll{});
    f.Do(action::mesh::Delete{action::mesh::DeleteMode::OnlyFaces});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {6u, 5u, 0u, 0u});
    expect(RenderedPrimitiveCounts(meshes, scene_id) == std::array{0u, 5u, 1u});
    f.P->Undo();
    f.Checkpoint();
    check_geometry(shift);
}

void TestMixedMeshletLod() {
    Fixture f{"mixed-meshlet-lod"};
    auto &meshes = f.R.Context.get<MeshStore>();
    MeshSource source;
    auto &data = source.Data;
    constexpr uint32_t Side = 32u, Width = Side + 1u, GridVertices = Width * Width;
    for (uint32_t y = 0u; y < Width; ++y)
        for (uint32_t x = 0u; x < Width; ++x)
            data.Positions.push_back({float(x) / Side, float(y) / Side, 0.f});
    for (uint32_t y = 0u; y < Side; ++y)
        for (uint32_t x = 0u; x < Side; ++x) {
            const auto v = y * Width + x;
            data.AddFace(std::array{v, v + 1u, v + Width});
            data.AddFace(std::array{v + 1u, v + Width + 1u, v + Width});
        }
    data.Positions.insert(data.Positions.end(), {{0, 0, 2}, {1, 0, 2}, {2, 0, 2}, {3, 0, 2}, {2, 1, 2}, {2, 0, 3}});
    data.Edges = {{0u, GridVertices}, {GridVertices, GridVertices + 1u}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    std::ranges::fill(meshes.EditFaceSharpness(id), 0u);
    std::ranges::fill(meshes.EditEdgeSharpness(id), 0u);
    mtl::ComputeChain chain{meshes.BufferContext()};
    meshes.UpdateCornerClassification(f.R, chain, std::array{id});
    chain.Submit();
    const auto [entity, instance] = AddMesh(f.R, id, MeshInstanceCreateInfo{});
    f.P->Settle();
    BuildMeshlets(f.R, chain, std::array{entity}, {});
    expect(BuildDemandedClusterLods(f.R, f.Viewport));
    expect(meshes.ClusterGroupCount(meshes.Get(id)) > 0u);
    const auto counts = RenderedPrimitiveCounts(meshes, id);
    expect(counts[1] == 2u);
    expect(counts[2] == 4u);
    f.Do(action::selection::Select{instance});
    f.EnterEdit(Element::Vertex);
    const std::array selected{GridVertices + 2u, GridVertices + 3u, GridVertices + 4u, GridVertices + 5u};
    f.SelectElements(entity, selected, Element::Vertex);
    f.P->History.Commit("Select hull vertices", {});
    const auto before = CountsOf(f.ActiveMesh());
    f.Do(action::mesh::ConvexHull{});
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {before.Vertices, before.Edges + 6u, before.Faces + 4u, before.Triangles + 4u});
    const auto after = RenderedPrimitiveCounts(meshes, id);
    expect(after[1] == 2u);
    expect(after[2] == 0u);
    f.P->Undo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), before);
    expect(RenderedPrimitiveCounts(meshes, id)[2] == 4u);
}

void TestAbsentFaceAttributes() {
    Fixture f{"absent-face-attributes"};
    f.Cube(Element::Face);
    f.Do(action::selection::SelectAll{});
    const auto history = *f.P->History.Present;
    f.Do(action::mesh::RotateColors{});
    f.Do(action::mesh::ReverseColors{});
    f.Do(action::mesh::RotateUVs{3u});
    f.Do(action::mesh::ReverseUVs{3u});
    f.Checkpoint();
    expect(*f.P->History.Present == history) << "absent layers must not create history";
}

void TestConnectVertices() {
    Fixture f{"connect-vertices"};
    MeshSource source;
    source.Data.Positions = {{2, 0, 0}, {1, 2, 0}, {-1, 2, 0}, {-2, 0, 0}, {-1, -2, 0}, {1, -2, 0}};
    source.Data.AddFace(std::array{0u, 1u, 2u, 3u, 4u, 5u});
    auto &uvs = source.Attrs.TexCoords0.emplace();
    for (const auto p : source.Data.Positions) uvs.push_back({p.x, p.y});
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = f.AddEditable(id, Element::Vertex);
    const auto original = Positions(f.ActiveMesh());
    f.SelectElements(entity, std::array{0u, 1u}, Element::Vertex);
    f.Do(action::mesh::ConnectVertices{});
    ExpectCounts(f.ActiveMesh(), {6u, 6u, 1u, 4u});
    f.SelectElements(entity, std::array{0u, 2u, 4u}, Element::Vertex);
    f.Do(action::mesh::ConnectVertices{});
    f.Checkpoint();
    const auto mesh = f.ActiveMesh();
    ExpectCounts(mesh, {6u, 9u, 4u, 4u});
    ExpectPositions(mesh, original);
    std::set<std::vector<uint32_t>> actual;
    for (const auto face : mesh.faces()) {
        std::vector<uint32_t> loop;
        for (const auto h : mesh.fh_range(face)) {
            const auto vertex = mesh.GetToVertex(h);
            loop.push_back(uint32_t(std::ranges::find(original, vertex, &VertexPositions::value_type::first) - original.begin()));
            const auto p = mesh.GetPosition(vertex);
            expect(f.R.Context.get<const MeshStore>().Arenas().CornerUvs[0].Get(*h) == vec2{p.x, p.y});
        }
        std::rotate(loop.begin(), std::ranges::min_element(loop), loop.end());
        actual.insert(std::move(loop));
    }
    expect(actual == std::set<std::vector<uint32_t>>{{0u, 1u, 2u}, {2u, 3u, 4u}, {0u, 4u, 5u}, {0u, 2u, 4u}});
    expect(f.Selection(Element::Face).Count() == 4u);
    expect(f.Selection(Element::Edge).Count() == 3u);
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1) boost::ut::cfg<> = {.filter = argv[1]};
    setvbuf(stdout, nullptr, _IONBF, 0);
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    boost::ut::suite tests = [] {
        using namespace boost::ut;
        "connect splits selected corners and preserves attributes"_test = TestConnectVertices;
        "mixed meshlet LOD preserves wire and point leaves"_test = TestMixedMeshletLod;
        "mixed meshlet builds group topologies and preserve clone ownership"_test = TestMixedMeshletBuild;
        "mixed mesh sources preserve wires attributes welding"_test = TestMixedMeshSources;
        "circularize fits selected boundaries"_test = TestCircularize;
        "curve fits loops between selected controls"_test = TestCurveBetweenSelected;
        "flatten projects independent selected regions"_test = TestFlatten;
        "edge relaxation matches alternating spline references"_test = TestRelaxEdgeLoops;
        "space evenly matches linear and cubic chain references"_test = TestSpaceEvenly;
        "grid fill preserves a mixed surface and wire boundary"_test = TestGridFill;
        "bridge joins mixed closed boundaries"_test = TestBridgeChains;
        "decimate reduces selected geometry and preserves boundaries attributes"_test = TestDecimate;
        "separate selection loose parts and materials preserve attributes render"_test = TestSeparate;
        "shrink fatten includes nonmanifold users and preserves cancelling normals"_test = TestShrinkFattenNonmanifold;
        "unsubdivide preserves wire selection and rejects unsafe boundaries"_test = TestUnsubdivideLinesAndBoundaries;
        "edit visibility matches Blender propagation selection"_test = TestEditVisibility;
        "hidden geometry exposes picks behind it and cannot be selected"_test = TestHiddenGeometryPicking;
        "visibility survives neighboring topology edits duplication and reopen"_test = TestVisibilityLifecycle;
        "snap symmetry updates partners and resolves overlapping matches"_test = TestSnapSymmetry;
        "unsubdivide matches Blender grids attributes"_test = TestUnsubdivide;
        "beautify faces matches Blender methods and converges"_test = TestBeautifyFaces;
        "beautify faces respects partial selection nonmanifold and existing edges"_test = TestBeautifyBoundaries;
        "beautify faces preserves seam provenance and relative winding"_test = TestBeautifySeams;
        "recalculate normals matches Blender components attributes"_test = TestRecalculateNormals;
        "radial edits preserve selection center"_test = TestRadialEdits;
        "radial edits match Blender across transformed meshes"_test = TestRadialTransformedMeshes;
        "shear matches transformed global and local axes and rejects invalid axes"_test = TestShearTransformedMeshes;
        "warp fits automatic bounds and continues outside explicit bounds"_test = TestWarpReferences;
        "bend uses the world plane and remains stable near zero angle"_test = TestBendReferences;
        "randomize preserves seeded samples distance controls normals"_test = TestRandomizeControls;
        "vertex slide matches Blender position controls"_test = TestVertexSlideReferences;
        "edge slide matches Blender position controls"_test = TestEdgeSlideReferences;
        "edge slide follows closed loops and rejects branches"_test = TestEdgeSlideClosedLoop;
        "edge slide follows Blender face paths and rejects nonmanifold edges"_test = TestEdgeSlideFacePaths;
        "vertex slide uses active reference and original endpoint positions"_test = TestVertexSlideReferenceAndDegeneracy;
        "slides preserve corner attributes and invalidate affected tangents through history"_test = TestSlideAttributes;
        "shrink fatten matches Blender face and vertex normal offsets"_test = TestShrinkFatten;
        "face attributes permute selected layers and preserve neighboring corners"_test = TestFaceAttributes;
        "face attributes skip absent layers"_test = TestAbsentFaceAttributes;
        "cube selection toggle moves selected vertices"_test = TestCubeSelectionMove;
        "face inset previews commit cancel and undo"_test = TestInsetGesture;
        "wire hull preserves nearby and unrelated wire connectivity"_test = TestWireHullPreservesConnectivity;
        "only faces preserves boundary and nonmanifold edges"_test = TestOnlyFacesPreservesEdges;
        "edge deletion preserves other edges"_test = TestDeleteEdgesPreservesOtherEdges;
        "duplicate and split preserve selected mixed geometry"_test = TestDuplicateGeometry;
        "vertex and face deletion preserve mixed connectivity"_test = TestDeleteVerticesAndFaces;
        "delete loose respects selection and preserves surface geometry"_test = TestDeleteLoose;
        "extrude region spin preserves corner attributes and authored normals"_test = TestExtrudeRegionSpinAttributes;
        "extrude region matches sequential mixed geometry edits"_test = TestExtrudeRegionMixed;
        "extrude edges creates faces from mixed geometry"_test = TestExtrudeEdgesMixed;
        "extrude vertices preserves mixed geometry and copies point attributes"_test = TestExtrudeVertices;
        "new edge and face reuse loose boundaries and preserve mixed geometry"_test = TestCreateGeometry;
        "tetrahedron hull returns to points after deleting faces"_test = TestPointHullDelete;
        "subdivide preserves mixed surface and wire geometry"_test = TestSubdivideMixed;
        "merge welds mixed surfaces wires and collapsed faces"_test = TestMergeMixed;
        "dissolve joins loose chains and preserves mixed surface connectivity"_test = TestDissolveMixed;
        "limited dissolve simplifies selected chains without erasing bends or closed loops"_test = TestLimitedDissolveChains;
        "limited dissolve respects vertex edge and face selections"_test = TestLimitedDissolveSelection;
        "limited dissolve respects material sharp and UV delimiters"_test = TestDissolveDelimiters;
        "dissolve controls preserve vertices and simplify all boundaries"_test = TestDissolveControls;
        "dissolve preserves the loose edges left by collapsed faces"_test = TestDissolveCollapsedFaces;
        "middle line edge subdivides into two rendered segments"_test = TestLineSubdivide;
        "smoothing respects selection iterations axis masks and isolated points"_test = TestSmoothVertices;
        "sparse selection survives block retirement undo and reopen"_test = TestSparseSelectionLifecycle;
        "planar faces flatten a saddle and leave triangles unchanged"_test = TestMakePlanarFaces;
        "wireframe quad positions attributes options"_test = TestWireframeQuad;
        "wireframe closed and partially selected regions"_test = TestWireframeRegion;
        "split nonplanar faces respects threshold attributes selection"_test = TestSplitNonplanarFaces;
        "split nonplanar concave faces uses interior diagonals in either winding"_test = TestSplitNonplanarConcave;
        "split nonplanar polygons have no fixed valence limit"_test = TestSplitNonplanarLargePolygon;
        "welding independent target groups preserves simple loops and loose edges"_test = TestWeldDistanceGroups;
        "welding duplicate faces preserves the surviving face and its attributes"_test = TestWeldDuplicateFaces;
        "position edits retessellate polygons through preview cancel commit"_test = TestPositionRetessellation;
        "concave fill and dissolve emit valid render triangles"_test = TestConcaveFillAndDissolve;
        "split concave faces matches Blender partitions attributes"_test = TestSplitConcaveFaces;
    };
    return RunSuites();
}
