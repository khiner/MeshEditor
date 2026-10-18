#include "Compress.h"
#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "animation/MorphWeights.h"
#include "armature/ArmatureComponents.h"
#include "editor/Engine.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "object/ObjectOps.h"
#include "project/Assets.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/SceneGraph.h"
#include "scene/SceneGraphOps.h"
#include "selection/SelectionGpu.h"
#include "selection/SelectionState.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderExtent.h"
#include "viewport/Viewport.h"

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
    auto Do(auto a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        return node.value();
    }
    void Commit(std::string label) {
        P->Settle();
        P->History.Commit(std::move(label), {});
        Audit();
    }
    state::Entity Add(auto a) {
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

// Verify action replay restores the document and a history branch, and workspace restoration restores the editor camera.
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

// Apply the parent transform once to a selected descendant by excluding the descendant from transform roots.
void TestSelectedGrandchildMovesOnce() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-grandchild"};
    Fixture f{dir};
    const Chain chain{f};
    expect(f.R.get<const TransformRoots>(f.Viewport).Roots == std::vector{chain.A});
    f.Do(action::view::TransformSelection{Transform{.P = {1, 0, 0}}});
    expect(std::abs(WorldTransformOf(f.R, chain.A)->P.x - 1.f) < 1e-4f);
    expect(std::abs(WorldTransformOf(f.R, chain.C)->P.x - 1.f) < 1e-4f);
}

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

// An unlinked duplicate copies its source's canonical values and render records onto storage of its own, so it hashes equal to its source.
// With its source's vertices turned in edit mode, which keeps their bounds, and the source hidden, the copy draws the image its source drew.
void TestDuplicateDrawsAsItsSource() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-duplicate-records"};
    Fixture f{dir};
    auto &r = f.R;
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto render_owner = [&](state::Entity e) -> const MeshStore::Record & {
        return meshes.Get(r.get<const MeshHandle>(r.get<const Instance>(e).Entity).StoreId);
    };
    const auto duplicate = [&](state::Entity source) {
        f.Do(action::selection::Select{source});
        const auto copy = f.Add(action::object::Duplicate{});
        expect(copy != source);
        expect(render_owner(copy).StoreId != render_owner(source).StoreId);
        expect(meshes.CanonicalHash(render_owner(copy).StoreId) == meshes.CanonicalHash(render_owner(source).StoreId));
        return copy;
    };
    const auto add_source = [&](MeshSource source) {
        const auto id = CreateMesh(r, std::move(source)).StoreId;
        const auto instance = ::AddMesh(r, id, MeshInstanceCreateInfo{}).second;
        f.P->Settle();
        f.Audit();
        return instance;
    };
    const auto fragment = [&](state::Entity instance, std::span<const uint32_t> selected) {
        f.Do(action::selection::Select{instance});
        f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
        f.Do(action::view::SetEditMode{.Mode = Element::Vertex});
        const auto entity = r.get<const Instance>(instance).Entity;
        ApplyEditSelectionLists(r, std::array{std::pair{entity, selected}}, Element::Vertex);
        f.Commit("Select clone holes");
        f.Do(action::mesh::Delete{action::mesh::DeleteMode::Vertices});
        f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    };
    MeshSource points;
    for (uint32_t i = 0u; i < 770u; ++i) points.Data.Positions.push_back({float(i) * .001f, float(i % 7u) * .1f, 0.f});
    const auto point_source = add_source(std::move(points));
    std::vector<uint32_t> removed;
    for (uint32_t i = 0u; i < 256u; ++i) removed.push_back(i);
    removed.push_back(512u);
    fragment(point_source, removed);
    expect(render_owner(duplicate(point_source)).RenderTopologies == 4u);
    MeshSource lines;
    for (uint32_t i = 0u; i < 520u; ++i) {
        lines.Data.Positions.push_back({float(i) * .001f, float(i % 3u) * .1f, 0.f});
        if (i) lines.Data.Edges.push_back({i - 1u, i});
    }
    const auto line_source = add_source(std::move(lines));
    fragment(line_source, std::array{255u, 300u});
    expect(render_owner(duplicate(line_source)).RenderTopologies == 2u);

    // Enough triangles to take a cluster hierarchy, with 128-face fans at the poles.
    const auto source = f.Add(action::object::AddMeshPrimitive{primitive::UVSphere{.Slices = 128u, .Stacks = 64u}, std::make_unique<MeshInstanceCreateInfo>()});
    expect(meshes.ClusterGroupCount(render_owner(source)) > 0u);
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    f.Do(action::view::SetEditMode{.Mode = Element::Face});
    const auto mesh_entity = r.get<const Instance>(source).Entity;
    const std::array selected_faces{2000u, 3000u};
    ApplyEditSelectionLists(r, std::array{std::pair{mesh_entity, std::span<const uint32_t>{selected_faces}}}, Element::Face);
    f.Commit("Select clone subdivisions");
    f.Do(action::mesh::Subdivide{2u});
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
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
    f.Do(action::selection::Select{point_source});
    f.Do(action::selection::ExtendActive{line_source});
    f.Do(action::selection::ExtendActive{copy});
    const auto mesh_count = r.view<const MeshHandle>().size();
    f.Do(action::object::Duplicate{});
    expect(r.view<const MeshHandle>().size() == mesh_count + 3u);
    f.Audit();
    f.P->Undo();
    expect(r.view<const MeshHandle>().size() == mesh_count);
    f.Audit();
    f.P->Redo();
    expect(r.view<const MeshHandle>().size() == mesh_count + 3u);
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

// Recompute layout expectations independently from records, instances, and mode.
// Verify disjoint bounds storage and compare topology and sharp-edge totals with a recount.
void CheckLayout(Fixture &f, std::string_view step) {
    auto &r = f.R;
    const auto &buffers = r.Context.get<const GpuBuffers>();
    const auto &meshes = r.Context.get<const MeshStore>();
    const auto &render = meshes.Render();
    const bool edit = r.get<const Interaction>(f.Viewport).Mode == InteractionMode::Edit;
    const bool transforming = edit && r.all_of<PendingTransform>(f.Viewport);
    const auto &primaries = r.get<const EditPrimaries>(f.Viewport).All;
    const auto fail = [&](std::string_view what) {
        std::printf("layout after %.*s: %.*s\n", int(step.size()), step.data(), int(what.size()), what.data());
        expect(false);
    };
    expect(r.view<const MeshLayoutDirty>().empty());
    std::array<std::vector<Range>, 3> runs;
    std::array<uint32_t, 3> topology_meshes{};
    uint32_t sharp_meshes = 0u;
    for (const auto [entity, models] : r.view<const ModelsBuffer>().each()) {
        const auto *record = TryRecordOf(r, entity);
        if (!record) continue;
        const auto states = buffers.Instances.StateBuffer.GetSpan<uint8_t>({models.InstanceRange.Offset, models.InstanceCount});
        if (std::ranges::any_of(states, [](uint8_t state) { return (state & InstanceStateHidden) == 0u; }) && meshes.MeshletCount(*record) > 0u) {
            for (uint32_t topology = 0u; topology < 3u; ++topology) topology_meshes[topology] += (record->RenderTopologies >> topology) & 1u;
            const auto flags = record->RenderTopologies ? render.MeshRecords.Get({record->StoreId, 1u})[0].Display.Flags : 0u;
            if ((flags & uint32_t(MeshletInstanceFlag::EditOverlay)) && meshes.GetEdgeSharpnessSummary(record->StoreId).Any) ++sharp_meshes;
        }
        if (!HasMesh(r, entity)) continue;
        const auto *run = r.try_get<const BoundsRun>(entity);
        if (!run) {
            fail("a mesh with instances has no bounds run");
            continue;
        }
        // Outside Edit mode, instances bound to an armature or morph weights deform apart.
        bool deformed = false;
        if (!edit) {
            for (const auto id : buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({models.InstanceRange.Offset, models.InstanceCount})) {
                const auto instance = r.EntityAt(ObjectIndex(id));
                const auto *modifier = record->SkinBlocksReady ? r.try_get<const ArmatureModifier>(instance) : nullptr;
                const auto *pose_state = modifier ? r.try_get<const ArmaturePoseState>(modifier->ArmatureEntity) : nullptr;
                deformed |= pose_state && modifier->SkinSlot < pose_state->GpuDeformRanges.size();
                deformed |= record->MorphBlocksReady && r.all_of<MorphWeightRange>(instance);
            }
        }
        const auto *work = r.try_get<const MeshEditWork>(entity);
        const bool posed = models.InstanceCount > 0u && (deformed || (edit && ((transforming && primaries.contains(entity)) || (work && work->RequiresPose))));
        const uint32_t count = deformed ? models.InstanceCount : 1u;
        if (run->Entries.Count != count) fail("entry count");
        if (run->Posed() != posed || r.all_of<PosedLayout>(entity) != posed) fail("posedness");
        const auto entries = buffers.BoundsReduceEntries.Get(run->Entries);
        for (uint32_t i = 0u; i < entries.size(); ++i) {
            if (entries[i].FirstInstance != models.InstanceRange.Offset + (deformed ? i : 0u) || entries[i].InstanceCount != (deformed ? 1u : models.InstanceCount)) fail("entry instances");
            if (entries[i].VertexOwner != record->Vertices.Index) fail("entry vertex owner");
        }
        runs[0].push_back(run->Entries);
        runs[1].push_back(run->DeriveEntries);
        runs[2].push_back(run->Tiles);
        if (!posed) {
            if (run->DeriveEntries.Count || run->Tiles.Count) fail("a static run holds derive entries or tile words");
            continue;
        }
        const auto &a = meshes.Arenas();
        const bool authored_morph = r.get<const PosedLayout>(entity).MorphDeformOffset != InvalidOffset && record->MorphShadingAuthored;
        const bool derive = !authored_morph && MakeDeriveEntryInputs(meshes, record->StoreId).has_value();
        uint32_t jobs = 0u;
        meshes.ForEachPrimitive(*record, [&](ElementHandle, const PrimitiveRecord &primitive) {
            if (primitive.LodFinestNode == InvalidOffset) return;
            const auto &node = render.LodNodes.Get({render.LodNodes.Address(record->NodeData, ElementHandle{primitive.LodFinestNode}).Value(), 1u})[0];
            jobs += node.Members != InvalidOffset && node.MeshletCount;
        });
        if (run->DeriveEntries.Count != (derive ? count : 0u)) fail("derive entry count");
        if (run->FaceBlocks != (derive ? a.FaceTriangles.Set(record->FaceData).BlockCount : 0u) || run->VertexBlocks != (derive ? a.Vertices.Set(record->Vertices).BlockCount : 0u)) fail("derive blocks");
        if (run->MeshletJobs != jobs) fail("posed meshlet jobs");
        if (run->BoundsKeys[0] != a.Vertices.Set(record->Vertices).BlockCount || run->BoundsKeys[3] != 1u) fail("bounds keys");
        const auto words = std::accumulate(run->BoundsKeys.begin(), run->BoundsKeys.end(), 0u) + run->FaceBlocks + run->VertexBlocks + 2u * run->MeshletJobs;
        if (run->Tiles.Count != words) fail("tile words");
    }
    for (auto &ranges : runs) {
        std::erase_if(ranges, [](Range range) { return range.Count == 0u; });
        std::ranges::sort(ranges, {}, &Range::Offset);
        for (uint32_t i = 1u; i < ranges.size(); ++i) {
            if (ranges[i - 1u].Offset + ranges[i - 1u].Count > ranges[i].Offset) fail("overlapping runs");
        }
    }
    if (buffers.TopologyMeshes != topology_meshes) fail("topology tallies");
    if (buffers.SharpEditMeshes != sharp_meshes) fail("sharp-edge tally");
}

// Verify mesh layout updates only when mesh inputs change across scene load, history, transforms, separation, and shading.
void TestIncrementalLayout() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-incremental-layout"};
    Fixture f{dir};
    auto &r = f.R;
    f.Do(action::io::LoadGltf{std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/RiggedSimple/glTF/RiggedSimple.gltf"});
    expect(!r.view<const PosedLayout>().empty());
    CheckLayout(f, "load");
    // A linked duplicate of the skinned object deforms apart from its source.
    const auto skinned = *r.view<const ArmatureModifier>().begin();
    f.Do(action::selection::Select{skinned});
    f.Add(action::object::DuplicateLinked{});
    expect(r.get<const BoundsRun>(r.get<const Instance>(skinned).Entity).Entries.Count == 2u);
    CheckLayout(f, "duplicate skinned");
    const auto cube = f.AddCube();
    CheckLayout(f, "add");
    const auto copy = f.Add(action::object::Duplicate{});
    CheckLayout(f, "duplicate");
    f.Do(action::object::SetSelectedVisible{false});
    CheckLayout(f, "hide");
    f.Do(action::object::Delete{});
    expect(!r.valid(copy));
    CheckLayout(f, "delete");
    f.P->Undo();
    CheckLayout(f, "undo");
    f.Do(action::selection::Select{cube});
    f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
    f.Do(action::view::SetEditMode{.Mode = Element::Face});
    f.Do(action::selection::SelectAll{});
    CheckLayout(f, "edit");
    action::Emit(action::view::TransformElements{{.P = {0.f, 1.f, 0.f}}}, action::Phase::Stage);
    f.P->Frame(action::Drain());
    expect(r.all_of<PosedLayout>(r.get<const Instance>(cube).Entity));
    CheckLayout(f, "drag");
    action::Commit();
    f.P->Frame(action::Drain());
    CheckLayout(f, "commit");
    const auto mesh_entity = r.get<const Instance>(cube).Entity;
    const std::array faces{0u, 1u};
    ApplyEditSelectionLists(r, std::array{std::pair{mesh_entity, std::span<const uint32_t>{faces}}}, Element::Face);
    f.Commit("Select faces");
    const auto mesh_count = r.view<const MeshHandle>().size();
    f.Do(action::mesh::Separate{});
    expect(r.view<const MeshHandle>().size() == mesh_count + 1u);
    CheckLayout(f, "separate");
    f.Do(action::view::SetInteractionMode{InteractionMode::Object});
    CheckLayout(f, "object");
    f.Do(action::object::SetSelectedSmoothShading{true});
    CheckLayout(f, "smooth");
    f.Do(action::view::SetViewportShading{ViewportShadingMode::Wireframe});
    CheckLayout(f, "wireframe");
}

} // namespace

void TestProcedures();

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
    TestIncrementalLayout();
    TestProcedures();
    return RunSuites();
}
