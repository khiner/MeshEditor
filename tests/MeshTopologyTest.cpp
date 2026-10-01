// Exercises four small mesh edits through production actions and viewport rendering.
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "action/Emit.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "metal/MetalCpp.h"
#include "numeric/MatrixMath.h"
#include "object/ObjectOps.h"
#include "project/Project.h"
#include "render/GpuBuffers.h"
#include "scene/Entity.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportDisplay.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
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
    for (const auto &[vertex, position] : expected) expect(Near(mesh.GetPosition(vertex), position));
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

    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
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
    const auto &buffers = f.R.Context.get<const GpuBuffers>();
    const auto &owner = buffers.MeshOf(id);
    expect(owner.RenderTopology == topology);
    expect(buffers.MeshletCount(owner) > 0u);
    uint32_t actual = 0u;
    buffers.ActiveMeshlets.ForEach(owner.MeshletRoot, [&](uint32_t handle) {
        const auto &meshlet = buffers.Meshlets.Get({handle, 1u})[0];
        expect(meshlet.Topology == topology);
        actual += meshlet.TriangleCount;
    });
    expect(actual == primitives);
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
}

void TestPointHullDelete() {
    Fixture f{"point-hull-delete"};
    MeshSource source;
    source.Data.Positions = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
    const auto id = CreateMesh(f.R, std::move(source)).StoreId;
    const auto [entity, instance] = AddMesh(f.R, id, MeshInstanceCreateInfo{});
    f.P->Settle();
    f.Do(action::selection::Select{instance});
    f.EnterEdit(Element::Vertex);
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
    ExpectCounts(f.ActiveMesh(), {4u, 0u, 0u, 0u});
    ExpectPositions(f.ActiveMesh(), original);
    ExpectRenderTopology(f, id, 2u, 4u);

    f.P->Undo();
    f.Checkpoint();
    ExpectCounts(f.ActiveMesh(), {4u, 6u, 4u, 4u});
    ExpectPositions(f.ActiveMesh(), original);
    ExpectRenderTopology(f, id, 0u, 4u);
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
} // namespace

int main(int argc, char **argv) {
    if (argc > 1) boost::ut::cfg<> = {.filter = argv[1]};
    setvbuf(stdout, nullptr, _IONBF, 0);
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    boost::ut::suite tests = [] {
        const auto pooled = [](auto test) {
            return [test] {
                const auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
                test();
            };
        };
        using namespace boost::ut;
        "cube selection toggle moves selected vertices"_test = pooled(TestCubeSelectionMove);
        "face inset previews commit cancel and undo"_test = pooled(TestInsetGesture);
        "tetrahedron hull returns to points after deleting faces"_test = pooled(TestPointHullDelete);
        "middle line edge subdivides into two rendered segments"_test = pooled(TestLineSubdivide);
    };
    return RunSuites();
}
