#include "project/Project.h"
#include "Compress.h"
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStores.h"
#include "object/ObjectOps.h"
#include "project/Assets.h"
#include "render/GpuBuffers.h"
#include "render/Instance.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "selection/SelectionState.h"
#include "viewport/RenderExtent.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>

using boost::ut::expect;

namespace {
bool Render = true;

struct Fixture : Engine {
    Fixture() : Engine{true} {
        R.Context.get<ViewportExtent>().Value = {64, 64};
    }
    explicit Fixture(const TestDir &dir) : Fixture{} { expect(P->New(dir)); }
    std::vector<std::byte> Workspace() {
        return workspace::Serialize(workspace::Capture(R, Viewport, R.Context.emplace<WindowsState>()));
    }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::printf("audit: %s\n", why.c_str());
        expect(valid);
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    template<typename A> auto Do(A a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        expect(R.Context.get<action::Errors>().Messages.empty());
        return node.value();
    }
    // Settles direct scene writes and commits them as one history node.
    void Commit(std::string label) {
        P->Settle();
        P->History.Commit(std::move(label), {});
    }
    // Does an action that adds an object and returns the object, selected and active.
    template<typename A> state::Entity Add(A a) {
        Do(std::move(a));
        return FindActiveEntity(R);
    }
    state::Entity AddCube() { return Add(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()}); }
    state::Entity AddEmpty(std::string name) {
        return Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Name = std::move(name)})});
    }
    std::vector<std::byte> Image() {
        SubmitViewport(R, Viewport);
        WaitForRender(R);
        const auto &image = R.Context.get<const RenderTargets>().Resources->FinalColorImage;
        return ReadbackImageRgba8(R.Context.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};

struct State {
    std::vector<std::byte> Persistent, Image;
    explicit State(Fixture &f)
        : Persistent(f.P->History.MaterializeLive()), Image(Render ? f.Image() : std::vector<std::byte>{}) {}
    void Check(Fixture &f) const {
        expect(f.P->History.MaterializeLive() == Persistent);
        if (Render) expect(f.Image() == Image);
        f.Audit();
    }
};

// A generated document restores its edited mesh and owned name through history and a saved project.
void TestSavedProject() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-basic"}, saved{"/tmp/mesheditor-scratch/project-basic-saved"};
    Fixture f{dir};
    auto &p = *f.P;
    const auto generated = f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto entity = FindActiveEntity(f.R);
    const auto original_name = f.R.get<Name>(entity).Value;
    const auto original = State{f};
    const auto moved = f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{1, 0, 0}));
    const auto transformed = State{f};
    const std::string renamed{"Generated cube with an owned name"};
    f.R.patch<Name>(entity, [&](auto &name) { name.Value = renamed; });
    const auto named = p.History.Commit("Rename generated cube", {});
    const auto edited = State{f};

    p.Undo();
    expect(f.R.get<Name>(entity).Value == original_name);
    transformed.Check(f);
    p.Redo();
    expect(f.R.get<Name>(entity).Value == renamed);
    edited.Check(f);
    p.Navigate(0);
    p.History.Evict(0);
    p.Navigate(generated);
    original.Check(f);
    p.Navigate(moved);
    transformed.Check(f);
    p.Navigate(named);
    expect(f.R.get<Name>(entity).Value == renamed);
    edited.Check(f);

    const auto workspace = f.Workspace();
    expect(p.SaveAs(saved, workspace));
    expect(p.Close());
    Fixture reopened;
    expect(reopened.P->Open(saved.Path / "working", saved.Path / "Saved.project"));
    expect(reopened.P->History.Present == named);
    expect(reopened.P->RestoredWorkspace == workspace);
    expect(reopened.R.get<Name>(entity).Value == renamed);
    edited.Check(reopened);
}

// Actions replay restores the present document and one branch while the final workspace restores the editor camera.
void TestActionsArchive() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-actions"}, archives{"/tmp/mesheditor-scratch/project-actions-archives"}, opened{"/tmp/mesheditor-scratch/project-actions-opened"};
    const auto archive = archives.Path / "Scene.actions";
    Fixture f{dir};
    auto &p = *f.P;
    const auto base = f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto present = f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{1, 0, 0}));
    const auto present_state = p.History.MaterializeLive();
    p.Navigate(base);
    const auto branch = f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{0, 2, 0}));
    const auto branch_state = p.History.MaterializeLive();
    p.Navigate(present);

    const CameraLens lens{Perspective{.FieldOfViewRad = .9f, .FarClip = 1000.f, .NearClip = .1f}};
    const ViewCamera editor_view{vec3{2, 3, 9}, vec3{0}, lens};
    f.R.replace<ViewCamera>(f.Viewport, editor_view);
    const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
    const auto final_node = f.Do(action::view::SetLookThroughCamera{camera});
    expect(LookThroughCameraEntity(f.R) == camera);
    expect(GetViewCameraState(f.R, f.Viewport).LookThroughSaved.value() == editor_view);
    const auto final_state = p.History.MaterializeLive();
    const auto final_camera = GetViewCameraState(f.R, f.Viewport);
    auto &windows = f.R.Context.emplace<WindowsState>();
    windows.History.Visible = false;
    windows.PendingTabs = {{11, 12}};
    const auto final_workspace = workspace::Serialize(workspace::Capture(f.R, f.Viewport, windows));
    expect(p.SaveArchive(archive, project::ArchiveForm::Actions, final_workspace));
    expect(ReadArchiveMetadata(archive).value() == final_workspace);

    expect(Decompress(archive, opened.Path));
    Fixture restored;
    expect(restored.P->Open(opened));
    expect(restored.P->History.Present == final_node);
    expect(restored.P->History.MaterializeLive() == final_state);
    const auto workspace = workspace::Deserialize(ReadArchiveMetadata(archive).value());
    expect(workspace.has_value());
    if (!workspace) return;
    auto &restored_windows = restored.R.Context.emplace<WindowsState>();
    workspace::Apply(restored.R, restored.Viewport, restored_windows, *workspace);
    expect(workspace::Serialize(workspace::Capture(restored.R, restored.Viewport, restored_windows)) == final_workspace);
    expect(GetViewCameraState(restored.R, restored.Viewport).Active == final_camera.Active);
    expect(GetViewCameraState(restored.R, restored.Viewport).LookThroughSaved == final_camera.LookThroughSaved);
    expect(LookThroughCameraEntity(restored.R) == camera);
    restored.Audit();

    restored.P->Navigate(branch);
    expect(restored.P->History.Present == branch);
    expect(restored.P->History.MaterializeLive() == branch_state);
    restored.Audit();
    restored.P->Navigate(present);
    expect(restored.P->History.MaterializeLive() == present_state);
    restored.P->Undo();
    expect(restored.P->History.Present == base);
    restored.P->Redo();
    expect(restored.P->History.Present == present);
    expect(restored.P->History.MaterializeLive() == present_state);
    restored.P->Navigate(final_node);
    expect(restored.P->History.MaterializeLive() == final_state);
    workspace::Apply(restored.R, restored.Viewport, restored_windows, *workspace);
    expect(GetViewCameraState(restored.R, restored.Viewport).Active == final_camera.Active);
    expect(GetViewCameraState(restored.R, restored.Viewport).LookThroughSaved == final_camera.LookThroughSaved);
    restored.Audit();
}

void WriteFile(const std::filesystem::path &path, std::string_view text) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream{path, std::ios::binary} << text;
}

// Writes the positions of the triangle (0,0,0), (x,0,0), (0,1,0) as a glTF buffer.
void WriteTriangle(const std::filesystem::path &path, float x) {
    const std::array<vec3, 3> positions{vec3{0, 0, 0}, vec3{x, 0, 0}, vec3{0, 1, 0}};
    WriteFile(path, {reinterpret_cast<const char *>(positions.data()), sizeof(positions)});
}

// External source references replay current contents and preserve the document when a dependency is missing.
void TestExternalReferences() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-external"}, source{"/tmp/mesheditor-scratch/project-external-source"}, archives{"/tmp/mesheditor-scratch/project-external-archives"}, opened{"/tmp/mesheditor-scratch/project-external-opened"};
    const auto gltf = source.Path / "Triangle.gltf", buffer = source.Path / "Triangle.bin", archive = archives.Path / "Triangle.actions";
    WriteFile(gltf, R"({
  "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [2, 1, 0]}],
  "bufferViews": [{"buffer": 0, "byteLength": 36}], "buffers": [{"byteLength": 36, "uri": "Triangle.bin"}]
})");
    const auto recorded_path = [](Fixture &f) {
        const auto &actions = f.P->DraftOf(*f.P->History.Present).RecordedActions;
        return std::get<action::io::LoadGltf>(std::get<action::io::Action>(actions.front().Action)).Path;
    };
    const auto max_x = [](Fixture &f) {
        expect(f.R.view<const MeshHandle>().size() == 1u);
        const auto mesh = GetMesh(f.R, *f.R.view<const MeshHandle>().begin());
        float largest = 0;
        for (uint32_t i = 0; i < mesh.VertexCount(); ++i) largest = std::max(largest, mesh.GetPosition(mesh.VertexAt(i)).x);
        return largest;
    };
    WriteTriangle(buffer, 1.f);
    Fixture f{dir};
    f.Do(action::io::LoadGltf{gltf});
    expect(recorded_path(f) == gltf && recorded_path(f).is_absolute());
    expect(max_x(f) == 1.f);
    expect(!std::filesystem::exists(f.P->History.Dir / project::Assets::DirectoryName));
    expect(f.P->SaveArchive(archive, project::ArchiveForm::Actions, f.Workspace()));
    expect(Decompress(archive, opened.Path));
    expect(!std::filesystem::exists(opened.Path / project::Assets::DirectoryName));

    WriteTriangle(buffer, 2.f);
    Fixture restored;
    expect(restored.P->Open(opened));
    expect(recorded_path(restored) == gltf);
    expect(max_x(restored) == 2.f);
    expect(restored.P->Replay());
    expect(max_x(restored) == 2.f);
    restored.Audit();
    const auto before_failure = restored.P->History.MaterializeLive();
    std::filesystem::remove(buffer);
    expect(!restored.P->Replay());
    expect(restored.P->History.MaterializeLive() == before_failure);
    auto &errors = restored.R.Context.get<action::Errors>().Messages;
    expect(!errors.empty());
    errors.clear();
    restored.Audit();
}

// The settle pass derives a RenderInstance for every Instance and writes Hidden into its state bit in place, through actions and history.
void TestRenderInstanceDerivation() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-render-instance"};
    Fixture f{dir};
    auto &p = *f.P;
    auto &r = f.R;
    state::DirtySet created;
    created.bind(r);
    created.on<RenderInstance>(state::On::Create | state::On::Destroy);
    const auto entity = f.AddCube();
    const auto slot = r.get<const RenderInstance>(entity).BufferIndex;
    const auto placed = [&](state::Entity e, bool hidden) {
        const auto *ri = r.try_get<const RenderInstance>(e);
        return ri && r.all_of<Hidden>(e) == hidden && ((r.Context.get<const GpuBuffers>().Instances.StateBuffer.GetSpan<uint8_t>()[ri->BufferIndex] & InstanceStateHidden) != 0u) == hidden;
    };
    expect(placed(entity, false));
    created.clear();
    f.Do(action::object::SetSelectedVisible{false});
    expect(placed(entity, true) && r.get<const RenderInstance>(entity).BufferIndex == slot);
    f.Do(action::object::SetSelectedVisible{true});
    expect(placed(entity, false) && r.get<const RenderInstance>(entity).BufferIndex == slot);
    f.Do(action::object::SetSelectedVisible{false});
    expect(placed(entity, true));
    p.Undo();
    expect(placed(entity, false));
    p.Redo();
    expect(placed(entity, true));
    expect(created.empty());
    const auto duplicate = f.Add(action::object::Duplicate{});
    expect(duplicate != entity && r.all_of<Instance>(duplicate) && placed(duplicate, true) && created.contains(duplicate));
    f.Do(action::object::SetSelectedVisible{true});
    expect(placed(duplicate, false));
    const auto other_mesh = r.get<const Instance>(f.AddCube()).Entity;
    r.replace<Instance>(duplicate, other_mesh);
    f.Commit("Retarget duplicate");
    expect(r.get<const RenderInstance>(duplicate).Entity == other_mesh);
    f.Audit();
}

// A hidden object draws no pixel a pick can hit.
void TestHiddenObjectEscapesPick() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-hidden-pick"};
    Fixture f{dir};
    auto &r = f.R;
    const auto cube = f.AddCube();
    const RenderView view{r.get<const ViewCamera>(f.Viewport), RenderExtentPx(r)};
    const auto pick = [&] { f.P->Do(action::MakeAction(action::selection::Pick{.Mouse = {0.5f, 0.5f}, .Shift = false, .View = std::make_unique<RenderView>(view)})); };
    f.Do(action::selection::DeselectAll{});
    pick();
    expect(r.all_of<Selected>(cube));
    f.Do(action::object::SetSelectedVisible{false});
    f.Do(action::selection::DeselectAll{});
    pick();
    expect(r.view<const Selected>().empty());
}

// Three empties A, B and C, chained A > B > C, with A and C selected and A active.
struct Chain {
    state::Entity A, B, C;
    explicit Chain(Fixture &f) {
        auto &r = f.R;
        A = f.AddEmpty("A"), B = f.AddEmpty("B"), C = f.AddEmpty("C");
        SetParent(r, B, A);
        SetParent(r, C, B);
        r.clear<Selected, Active>();
        r.emplace<Selected>(A);
        r.emplace<Selected>(C);
        r.emplace<Active>(A);
        f.Commit("Hierarchy");
    }
};

// A selected grandchild of a selected object is no transform root, and moving the object moves the grandchild by the delta once.
void TestSelectedGrandchildMovesOnce() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-grandchild"};
    Fixture f{dir};
    const Chain chain{f};
    expect(f.R.get<const TransformRoots>(f.Viewport).Roots == std::vector{chain.A});
    f.Do(action::view::TransformSelection{Transform{.P = {1, 0, 0}}});
    expect(std::abs(WorldTransformOf(f.R, chain.A)->P.x - 1.f) < 1e-4f);
    expect(std::abs(WorldTransformOf(f.R, chain.C)->P.x - 1.f) < 1e-4f);
}

// Hiding, deleting and duplicating instances of a shared mesh and of another mesh, then undoing each, renders the scene's image again.
void TestInstanceLifecycleRendersBack() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-instance-lifecycle"};
    Fixture f{dir};
    auto &r = f.R;
    const auto add_cube = [&](float x) {
        const auto cube = f.AddCube();
        f.Do(action::view::TransformSelection{Transform{.P = {x, 0, 0}}});
        return cube;
    };
    const auto first = add_cube(-2.f);
    const auto linked = f.Add(action::object::DuplicateLinked{});
    f.Do(action::view::TransformSelection{Transform{.P = {2, 0, 0}}});
    const auto other = add_cube(4.f);
    expect(r.get<const Instance>(linked).Entity == r.get<const Instance>(first).Entity);
    // Runs the actions on the selected entity, then undoes them back to the image before them.
    const auto undone = [&](state::Entity selected, auto... leaves) {
        f.Do(action::selection::Select{selected});
        const State before{f};
        (f.Do(std::move(leaves)), ...);
        if (Render) expect(f.Image() != before.Image);
        for (size_t i = 0u; i < sizeof...(leaves); ++i) f.P->Undo();
        before.Check(f);
    };
    undone(first, action::object::SetSelectedVisible{false});
    undone(linked, action::object::Delete{});
    undone(other, action::object::DuplicateLinked{}, action::view::TransformSelection{Transform{.P = {0, 2, 0}}});
    undone(other, action::object::Delete{});
}

// An unlinked duplicate copies its source's render records onto its own geometry.
// The copy's finest clusters name its own triangles, edges or points, whose owner entries name those clusters.
// Each copied vertex fan holds its source fan's corner and face pairs, offset into the copy's arenas.
// With its source's vertices turned in edit mode, which keeps their bounds, and the source hidden, the copy draws the image its source drew.
void TestDuplicateDrawsAsItsSource() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-duplicate-records"};
    Fixture f{dir};
    auto &r = f.R;
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto render_owner = [&](state::Entity e) -> const MeshBuffers & {
        return buffers.MeshOf(r.get<const MeshHandle>(r.get<const Instance>(e).Entity).StoreId);
    };
    // Duplicates the selected source and returns the copy, checking that its render records are its own.
    const auto duplicate = [&](state::Entity source) {
        f.Do(action::selection::Select{source});
        const auto copy = f.Add(action::object::Duplicate{});
        expect(copy != source);
        const auto &copied = render_owner(copy);
        expect(buffers.MeshletCount(copied) == buffers.MeshletCount(render_owner(source)));
        expect(buffers.ClusterGroupCount(copied) == buffers.ClusterGroupCount(render_owner(source)));
        const auto topology = copied.RenderTopology;
        const auto &record = meshes.Get(copied.StoreId);
        const auto &a = meshes.Arenas();
        const auto first = topology == 0u ? a.Triangles.First(record.TriangleData) : topology == 1u ? a.EdgeHalfedges.First(record.EdgeData) : a.Vertices.First(record.Vertices);
        const auto count = topology == 0u ? a.Triangles.Count(record.TriangleData) : topology == 1u ? a.EdgeHalfedges.Count(record.EdgeData) : a.Vertices.Count(record.Vertices);
        const auto origin = topology == 0u ? 0u : copied.ElementMeshletOrigin;
        bool owned = true;
        buffers.ActiveMeshlets.ForEach(copied.MeshletRoot, [&](uint32_t id) {
            const auto &meshlet = buffers.Meshlets.Get({id, 1u})[0];
            if (meshlet.RefinedGroup != InvalidOffset) return;
            for (const auto element : buffers.MeshletTriangleIds.Get({meshlet.TriangleOffset, meshlet.TriangleCount})) {
                const auto handle = origin + element;
                owned &= handle >= first && handle < first + count && buffers.ElementMeshlets[topology].Get(handle) == id;
            }
        });
        expect(owned);
        const auto &source_record = meshes.Get(render_owner(source).StoreId);
        const auto corner_delta = a.FaceCorners.First(record.FaceCorners) - a.FaceCorners.First(source_record.FaceCorners);
        const auto face_delta = a.FaceTriangles.First(record.FaceData) - a.FaceTriangles.First(source_record.FaceData);
        const auto offset = [](uint32_t value, uint32_t delta) { return value == InvalidOffset ? value : value + delta; };
        const auto source_fans = a.VertexCorners.Get(a.Vertices.Dense(source_record.Vertices)), fans = a.VertexCorners.Get(a.Vertices.Dense(record.Vertices));
        const auto items = a.VertexFans.Items.Buffer.GetSpan<uvec2>();
        bool fans_copied = source_fans.size() == fans.size();
        for (size_t v = 0u; fans_copied && v < fans.size(); ++v) {
            fans_copied &= fans[v].y == source_fans[v].y;
            for (uint32_t i = 0u; fans_copied && i < fans[v].y; ++i) {
                const auto item = items[source_fans[v].x + i];
                fans_copied &= items[fans[v].x + i] == uvec2{offset(item.x, corner_delta), offset(item.y, face_delta)};
            }
        }
        expect(fans_copied);
        return copy;
    };
    const auto add_source = [&](MeshSource source) {
        const auto id = CreateMesh(r, std::move(source)).StoreId;
        const auto instance = ::AddMesh(r, id, MeshInstanceCreateInfo{}).second;
        f.P->Settle();
        return instance;
    };
    MeshSource points;
    points.Data.Positions = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
    expect(render_owner(duplicate(add_source(std::move(points)))).RenderTopology == 2u);
    MeshSource lines;
    lines.Data.Positions = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    lines.Data.Edges = {{0u, 1u}, {1u, 2u}};
    expect(render_owner(duplicate(add_source(std::move(lines)))).RenderTopology == 1u);

    // Enough triangles to take a cluster hierarchy, with 128-face fans at the poles.
    const auto source = f.Add(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = 128u, .Stacks = 64u}, std::make_unique<MeshInstanceCreateInfo>()});
    expect(buffers.ClusterGroupCount(render_owner(source)) > 0u);
    // Only the sphere draws in both images.
    f.Do(action::selection::SelectAll{});
    f.Do(action::object::SetSelectedVisible{false});
    f.Do(action::selection::Select{source});
    f.Do(action::object::SetSelectedVisible{true});
    const auto before = f.Image();
    const auto copy = duplicate(source);
    f.Do(action::selection::Select{source});
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    f.Do(action::selection::SelectAll{});
    action::Emit(action::view::TransformElements{{.R = AngleAxis(std::numbers::pi_v<float> / 2.f, vec3{1.f, 0.f, 0.f})}}, action::Phase::Stage);
    f.P->Frame(action::Drain());
    action::Commit();
    f.P->Frame(action::Drain());
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    f.Do(action::object::SetSelectedVisible{false});
    f.Do(action::selection::Select{copy});
    if (Render) expect(f.Image() == before);
    f.Audit();
}

// Parenting an object to its own descendant fails and leaves the hierarchy without a cycle.
void TestParentToDescendantFails() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-parent-cycle"};
    Fixture f{dir};
    auto &r = f.R;
    const Chain chain{f};
    f.Do(action::selection::ExtendActive{chain.C});
    f.P->Do(action::MakeAction(action::object::ParentToActive{}));
    auto &messages = r.Context.get<action::Errors>().Messages;
    expect(messages == std::vector<std::string>{"Loop in parents"});
    messages.clear();
    expect(ParentOrNull(r, chain.A) == state::Null);
    expect(ParentOrNull(r, chain.C) == chain.B);
    f.Audit();
}

} // namespace

int main(int argc, char **argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc == 2 && std::string_view{argv[1]} == "--no-render") Render = false;
    else if (argc != 1) {
        std::fprintf(stderr, "Usage: %s [--no-render]\n", argv[0]);
        return 1;
    }
    if (!Render) std::puts("Skipping rendered-image comparisons (--no-render).");
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    TestSavedProject();
    TestActionsArchive();
    TestExternalReferences();
    TestRenderInstanceDerivation();
    TestHiddenObjectEscapesPick();
    TestSelectedGrandchildMovesOnce();
    TestInstanceLifecycleRendersBack();
    TestDuplicateDrawsAsItsSource();
    TestParentToDescendantFails();
    return RunSuites();
}
