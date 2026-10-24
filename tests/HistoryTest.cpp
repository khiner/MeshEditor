#include "Compress.h"
#include "ProjectFixture.h"
#include "audio/AudioOutputConfig.h"
#include "gpu/WorkspaceLights.h"
#include "mesh/MeshComponents.h"
#include "scene/SceneGraph.h"
#include "snapshot/Workspace.h"
#include "viewport/InteractionComponents.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportDisplay.h"
#include "viewport/ViewportScene.h"

using boost::ut::expect;

namespace {
struct HistoryFixture : Fixture {
    using Fixture::Fixture;
    // Adds a viewport, which takes the highest index.
    state::Entity AddViewport() {
        Do(action::view::AddViewport{});
        return std::ranges::max(R.view<const ViewportScene>(), {}, state::Index);
    }
    ViewCameraState MoveCamera(state::Entity viewport, vec3 target) {
        Do(action::view::SetViewCameraTarget{viewport, target});
        const auto camera = Camera(viewport);
        expect(camera != ViewCameraState{});
        return camera;
    }
    ViewCameraState Camera(state::Entity viewport) const { return R.get<const ViewCameraState>(viewport); }
};

// The live application: the first viewport's rendered image, the document, and the workspace.
struct Live {
    std::vector<std::byte> Image, Document, Workspace;
    explicit Live(Fixture &f)
        : Image(Render ? f.Image() : std::vector<std::byte>{}), Document(f.P->History.MaterializeLive()), Workspace(snapshot::SerializeWorkspace(f.R)) {}
    void Check(Fixture &f) const {
        if (Render) expect(f.Image() == Image);
        expect(f.P->History.MaterializeLive() == Document);
        expect(snapshot::SerializeWorkspace(f.R) == Workspace);
        f.Audit();
    }
};

std::set<state::Entity> Viewports(const state::Registry &r) { return r.view<const ViewportScene>() | std::ranges::to<std::set>(); }

void ExpectDefaults(const state::Registry &r, state::Entity viewport) {
    [&]<typename... Cs>(TypeList<Cs...>) { (expect(r.get<const Cs>(viewport) == ViewDefault<Cs>()), ...); }(ViewValues{});
}

// The finished history reproduces at every node through an actions archive and at the present through a saved project, each with the workspace applied.
void CheckReopened(HistoryFixture &f, const TestDir &dir) {
    auto &p = *f.P;
    const auto present = p.History.Present.value();
    std::vector<Live> nodes;
    for (uint32_t node = 0; node < p.History.Nodes.size(); ++node) {
        p.Navigate(node);
        nodes.emplace_back(f);
    }
    p.Navigate(present);
    const auto &workspace = nodes[present].Workspace;

    const TestDir archives{(dir.Path.string() + "-archive").c_str()}, opened{(dir.Path.string() + "-opened").c_str()};
    const auto archive = archives.Path / "Scene.actions";
    expect(p.SaveArchive(archive, project::ArchiveForm::Actions, workspace));
    expect(ReadArchiveMetadata(archive).value() == workspace);
    expect(Decompress(archive, opened.Path));
    HistoryFixture restored;
    expect(restored.P->Open(opened));
    expect(restored.P->History.Present == present);
    // The replayed actions alone return every node's document.
    for (uint32_t node = 0; node < nodes.size(); ++node) {
        restored.P->Navigate(node);
        expect(restored.P->History.MaterializeLive() == nodes[node].Document);
        restored.Audit();
    }
    snapshot::ApplyWorkspace(restored.R, workspace);
    for (uint32_t node = 0; node < nodes.size(); ++node) {
        restored.P->Navigate(node);
        nodes[node].Check(restored);
    }

    const TestDir saved{(dir.Path.string() + "-saved").c_str()};
    const auto bytes = f.Workspace();
    expect(p.SaveAs(saved, bytes));
    expect(p.Close());
    HistoryFixture reopened;
    expect(reopened.P->Open(saved.Path / "working", saved.Path / "Saved.project"));
    expect(reopened.P->History.Present == present);
    expect(reopened.P->RestoredWorkspace == bytes);
    snapshot::ApplyWorkspace(reopened.R, workspace);
    nodes[present].Check(reopened);
}

// Runs `script`, which calls `step` after each action it takes.
// Every step reproduces through a hot restore, a cold restore, and a replay, a viewport a recorded step creates starts from defaults, and the finished history reopens.
void Scenario(const char *name, auto &&script) {
    boost::ut::test(name) = [&] {
        const TestDir dir{std::format("/tmp/mesheditor-scratch/history-{}", name).c_str()};
        HistoryFixture f{dir};
        auto &p = *f.P;
        auto viewports = Viewports(f.R);
        auto nodes = p.History.Nodes.size();
        const auto step = [&] {
            if (p.History.Nodes.size() > nodes) {
                for (const auto viewport : Viewports(f.R)) {
                    if (!viewports.contains(viewport)) ExpectDefaults(f.R, viewport);
                }
            }
            const Live live{f};
            if (p.History.UndoTarget()) {
                p.Undo();
                p.Redo();
                live.Check(f);
                const auto cap = std::exchange(p.MemoryCap, 0);
                p.Undo();
                expect(!p.History.Nodes[p.History.RedoTarget().value()].Hot);
                p.Redo();
                live.Check(f);
                p.MemoryCap = cap;
            }
            expect(p.Replay());
            live.Check(f);
            viewports = Viewports(f.R);
            nodes = p.History.Nodes.size();
        };
        script(f, step);
        CheckReopened(f, dir);
    };
}

// Switching documents starts every workspace value from defaults and keeps the output device setting.
// A record for an identity that is another kind of entity is retained rather than placed.
void TestDocumentSwitch() {
    const TestDir dir{"/tmp/mesheditor-scratch/history-switch"}, other_dir{"/tmp/mesheditor-scratch/history-switch-other"}, archives{"/tmp/mesheditor-scratch/history-switch-archive"}, opened{"/tmp/mesheditor-scratch/history-switch-opened"}, next{"/tmp/mesheditor-scratch/history-switch-next"};
    const auto archive = archives.Path / "Scene.actions";
    HistoryFixture other{other_dir};
    const auto viewport = other.AddViewport();
    const auto added = other.P->History.Present.value();
    other.P->Undo();
    const auto other_workspace = snapshot::SerializeWorkspace(other.R);
    expect(other.P->SaveArchive(archive, project::ArchiveForm::Actions, other_workspace));
    expect(Decompress(archive, opened.Path));

    HistoryFixture f{dir};
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
    f.P->Navigate(added);
    expect(f.Camera(viewport) == ViewCameraState{});
    expect(f.R.Resources.get<const AudioOutputConfig>().SampleRate == 96000u);

    f.P->Navigate(0);
    f.AddCube();
    expect(f.R.all_of<MeshHandle>(viewport));
    snapshot::ApplyWorkspace(f.R, other_workspace);
    expect(!f.R.all_of<ViewCameraState>(viewport));
    expect(snapshot::SerializeWorkspace(f.R) == other_workspace);
    f.Audit();
}
} // namespace

void TestHistory() {
    Scenario("viewport removal", [](HistoryFixture &f, auto step) {
        const auto viewport = f.AddViewport();
        step();
        f.MoveCamera(viewport, {1, 2, 3});
        step();
        f.Do(action::view::RemoveViewport{viewport});
        step();
        expect(!f.R.valid(viewport));
    });
    Scenario("fresh identity after an undone add", [](HistoryFixture &f, auto step) {
        const auto viewport = f.AddViewport();
        step();
        f.MoveCamera(viewport, {1, 2, 3});
        step();
        f.P->Undo();
        step();
        expect(f.AddViewport() == viewport);
        step();
    });
    Scenario("two generations at one index", [](HistoryFixture &f, auto step) {
        const auto first = f.AddViewport();
        step();
        f.MoveCamera(first, {1, 2, 3});
        step();
        f.Do(action::view::RemoveViewport{first});
        step();
        const auto second = f.AddViewport();
        step();
        expect(state::Index(second) == state::Index(first));
        expect(state::Generation(second) > state::Generation(first));
        f.MoveCamera(second, {-4, 5, -6});
        step();
    });
    Scenario("one identity as two kinds", [](HistoryFixture &f, auto step) {
        const auto viewport = f.AddViewport();
        step();
        f.MoveCamera(viewport, {1, 2, 3});
        step();
        f.P->Undo();
        step();
        f.AddCube();
        step();
        expect(f.R.all_of<MeshHandle>(viewport));
        expect(!f.R.any_of<ViewportScene, ViewCameraState, Viewport>(viewport));
    });
    Scenario("unique names", [](HistoryFixture &f, auto step) {
        const auto name = [&](state::Entity e) { return f.R.get<const Name>(e).Value; };
        f.AddEmpty("Thing");
        step();
        f.AddEmpty("Thing");
        step();
        expect(name(f.AddEmpty("Thing")) == "Thing_2");
        step();
        f.Do(action::object::Delete{});
        step();
        expect(name(f.AddEmpty("Thing")) == "Thing_2");
        step();
    });
    Scenario("edits and a branch", [](HistoryFixture &f, auto step) {
        f.AddCube();
        step();
        f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{1, 0, 0}));
        step();
        // A Workspace component's Update applies without recording.
        const auto viewport = FirstViewport(f.R);
        const auto nodes = f.P->History.Nodes.size();
        f.Do(action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, 2.f));
        step();
        expect(f.P->History.Nodes.size() == nodes);
        expect(f.R.get<const ViewportDisplay>(viewport).LodErrorPixels == 2.f);
        const CameraLens lens{Perspective{.FieldOfViewRad = .9f, .FarClip = 1000.f, .NearClip = .1f}};
        const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
        step();
        f.Do(action::view::SetLookThroughCamera{viewport, camera});
        step();
        expect(f.Camera(viewport).LookThrough == camera);
        f.P->Undo();
        step();
        f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{0, 2, 0}));
        step();
    });
    boost::ut::test("switching documents") = TestDocumentSwitch;
}
