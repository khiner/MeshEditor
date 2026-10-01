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
#include "mesh/MeshStores.h"
#include "project/Assets.h"
#include "render/RenderTargets.h"
#include "render/Textures.h"
#include "scene/Entity.h"
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
    template<typename A> int Do(A a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        expect(R.Context.get<action::Errors>().Messages.empty());
        return node;
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
    Fixture f;
    auto &p = *f.P;
    expect(p.New(dir));
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
    Fixture f;
    auto &p = *f.P;
    expect(p.New(dir));
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
    f.Do(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
    const auto camera = FindActiveEntity(f.R);
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
    restored.P->Navigate(final_node);
    expect(restored.P->History.MaterializeLive() == final_state);
    workspace::Apply(restored.R, restored.Viewport, restored_windows, *workspace);
    expect(GetViewCameraState(restored.R, restored.Viewport).Active == final_camera.Active);
    expect(GetViewCameraState(restored.R, restored.Viewport).LookThroughSaved == final_camera.LookThroughSaved);
    restored.Audit();
}

// External source references replay current contents and preserve the document when a dependency is missing.
void TestExternalReferences() {
    const TestDir dir{"/tmp/mesheditor-scratch/project-external"}, source{"/tmp/mesheditor-scratch/project-external-source"}, archives{"/tmp/mesheditor-scratch/project-external-archives"}, opened{"/tmp/mesheditor-scratch/project-external-opened"};
    std::filesystem::create_directories(source.Path);
    const auto gltf = source.Path / "Triangle.gltf", buffer = source.Path / "Triangle.bin", archive = archives.Path / "Triangle.actions";
    std::ofstream{gltf} << R"({
  "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
  "nodes": [{"mesh": 0}], "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
  "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [2, 1, 0]}],
  "bufferViews": [{"buffer": 0, "byteLength": 36}], "buffers": [{"byteLength": 36, "uri": "Triangle.bin"}]
})";
    const auto write_positions = [&](float x) {
        const std::array<vec3, 3> positions{vec3{0, 0, 0}, vec3{x, 0, 0}, vec3{0, 1, 0}};
        std::ofstream bin{buffer, std::ios::binary};
        bin.write(reinterpret_cast<const char *>(positions.data()), sizeof(positions));
    };
    const auto recorded_path = [](Fixture &f) {
        const auto &actions = f.P->DraftOf(f.P->History.Present).RecordedActions;
        return std::get<action::io::LoadGltf>(std::get<action::io::Action>(actions.front().Action)).Path;
    };
    const auto max_x = [](Fixture &f) {
        expect(f.R.view<const MeshHandle>().size() == 1u);
        const auto mesh = GetMesh(f.R, *f.R.view<const MeshHandle>().begin());
        float largest = 0;
        for (uint32_t i = 0; i < mesh.VertexCount(); ++i) largest = std::max(largest, mesh.GetPosition(Mesh::VH{i}).x);
        return largest;
    };
    write_positions(1.f);
    Fixture f;
    expect(f.P->New(dir));
    f.Do(action::io::LoadGltf{gltf});
    expect(recorded_path(f) == gltf && recorded_path(f).is_absolute());
    expect(max_x(f) == 1.f);
    expect(!std::filesystem::exists(f.P->History.Dir / project::Assets::DirectoryName));
    expect(f.P->SaveArchive(archive, project::ArchiveForm::Actions, f.Workspace()));
    expect(Decompress(archive, opened.Path));
    expect(!std::filesystem::exists(opened.Path / project::Assets::DirectoryName));

    write_positions(2.f);
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
    return RunSuites();
}
