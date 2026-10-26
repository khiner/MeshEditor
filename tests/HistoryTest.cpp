#include "ProjectFixture.h"
#include "audio/AudioOutputConfig.h"
#include "gpu/WorkspaceLights.h"
#include "scene/SceneGraph.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportScene.h"

using boost::ut::expect;

namespace {
struct HistoryFixture : Fixture {
    // Adds a viewport, which takes the highest index and starts from every view value's default.
    state::Entity AddViewport() {
        Do(action::view::AddViewport{});
        const auto viewport = std::ranges::max(R.view<const ViewportScene>(), {}, state::Index);
        [&]<typename... Cs>(TypeList<Cs...>) { (expect(R.get<const Cs>(viewport) == ViewDefault<Cs>()), ...); }(ViewValues{});
        return viewport;
    }
    ViewCameraState MoveCamera(state::Entity viewport, vec3 target) {
        Do(action::view::SetViewCameraTarget{viewport, target});
        const auto camera = Camera(viewport);
        expect(camera != ViewCameraState{});
        return camera;
    }
    ViewCameraState Camera(state::Entity viewport) const { return R.get<const ViewCameraState>(viewport); }
};

// Switching documents starts every workspace value from defaults and keeps the output device setting.
// A record for an identity that is another kind of entity is retained rather than placed.
void TestDocumentSwitch(HistoryFixture &f, HistoryFixture &other) {
    const TestDir dir{"/tmp/mesheditor-scratch/history-switch"}, other_dir{"/tmp/mesheditor-scratch/history-switch-other"}, archives{"/tmp/mesheditor-scratch/history-switch-archive"}, opened{"/tmp/mesheditor-scratch/history-switch-opened"}, next{"/tmp/mesheditor-scratch/history-switch-next"};
    const auto archive = archives.Path / "Scene.actions";
    other.Begin(other_dir.Path);
    const auto viewport = other.AddViewport();
    const auto added = other.P->History.Present.value();
    other.Undo();
    const auto other_workspace = snapshot::SerializeWorkspace(other.R);
    expect(other.P->SaveArchive(archive, project::ArchiveForm::Actions, other_workspace));
    expect(Decompress(archive, opened.Path));

    f.Begin(dir.Path);
    f.MoveCamera(FirstViewport(f.R), {1, 2, 3});
    f.R.replace<ActiveViewport>(state::Document, ActiveViewport{FirstViewport(f.R)});
    f.Do(action::audio::SetOutputDevice{.SampleRate = 96000u});
    expect(f.P->New(next));
    expect(f.Camera(FirstViewport(f.R)) == ViewCameraState{});
    expect(f.R.get<const ActiveViewport>(state::Document) == ActiveViewport{});
    expect(f.R.Resources.get<const AudioOutputConfig>().SampleRate == 96000u);

    f.MoveCamera(FirstViewport(f.R), {1, 2, 3});
    expect(f.AddViewport() == viewport);
    f.MoveCamera(viewport, {4, 5, 6});
    f.Do(action::view::RemoveViewport{viewport});
    expect(f.P->Open(opened));
    expect(f.Camera(FirstViewport(f.R)) == ViewCameraState{});
    f.Navigate(added);
    expect(f.Camera(viewport) == ViewCameraState{});
    expect(f.R.Resources.get<const AudioOutputConfig>().SampleRate == 96000u);

    f.Navigate(0);
    f.AddCube();
    expect(f.R.all_of<primitive::Cuboid>(viewport));
    snapshot::ApplyWorkspace(f.R, other_workspace);
    expect(!f.R.all_of<ViewCameraState>(viewport));
    expect(snapshot::SerializeWorkspace(f.R) == other_workspace);
    f.Audit();
}
} // namespace

void TestHistory() {
    HistoryFixture f, other;
    Scenario(f, other, "viewport removal", [](HistoryFixture &f) {
        const auto viewport = f.AddViewport();
        f.MoveCamera(viewport, {1, 2, 3});
        f.Do(action::view::RemoveViewport{viewport});
        expect(!f.R.valid(viewport));
    });
    Scenario(f, other, "fresh identity after an undone add", [](HistoryFixture &f) {
        const auto viewport = f.AddViewport();
        f.MoveCamera(viewport, {1, 2, 3});
        f.Undo();
        expect(f.AddViewport() == viewport);
    });
    Scenario(f, other, "two generations at one index", [](HistoryFixture &f) {
        const auto first = f.AddViewport();
        f.MoveCamera(first, {1, 2, 3});
        f.Do(action::view::RemoveViewport{first});
        const auto second = f.AddViewport();
        expect(state::Index(second) == state::Index(first));
        expect(state::Generation(second) > state::Generation(first));
        f.MoveCamera(second, {-4, 5, -6});
    });
    Scenario(f, other, "one identity as two kinds", [](HistoryFixture &f) {
        const auto viewport = f.AddViewport();
        f.MoveCamera(viewport, {1, 2, 3});
        f.Undo();
        f.AddCube();
        expect(f.R.all_of<primitive::Cuboid>(viewport));
        expect(!f.R.any_of<ViewportScene, ViewCameraState, Viewport>(viewport));
    });
    Scenario(f, other, "unique names", [](HistoryFixture &f) {
        const auto name = [&](state::Entity e) { return f.R.get<const Name>(e).Value; };
        f.AddEmpty("Thing");
        f.AddEmpty("Thing");
        expect(name(f.AddEmpty("Thing")) == "Thing_2");
        f.Do(action::object::Delete{});
        expect(name(f.AddEmpty("Thing")) == "Thing_2");
    });
    Scenario(f, other, "edits and a branch", [](HistoryFixture &f) {
        f.AddCube();
        f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{1, 0, 0}));
        // A Workspace component's Update applies without recording.
        const auto viewport = FirstViewport(f.R);
        const auto nodes = f.P->History.Nodes.size();
        f.Do(action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, 2.f));
        expect(f.P->History.Nodes.size() == nodes);
        expect(f.R.get<const ViewportDisplay>(viewport).LodErrorPixels == 2.f);
        const CameraLens lens{Perspective{.FieldOfViewRad = .9f, .FarClip = 1000.f, .NearClip = .1f}};
        const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
        f.Do(action::view::SetLookThroughCamera{viewport, camera});
        expect(LookThroughCamera(f.R, viewport) == camera);
        // Undoing the camera's creation leaves the viewport looking through it once it returns.
        f.Undo();
        expect(LookThroughCamera(f.R, viewport) == state::Null && f.Camera(viewport).LookThrough == camera);
        f.Redo();
        expect(LookThroughCamera(f.R, viewport) == camera);
        f.Undo();
        f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{0, 2, 0}));
    });
    boost::ut::test("switching documents") = [&] { TestDocumentSwitch(f, other); };
}
