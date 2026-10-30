#include "ProjectFixture.h"
#include "audio/AudioOutputConfig.h"
#include "project/HistoryUi.h"
#include "selection/SelectionGpu.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportScene.h"
#include <imgui_internal.h>

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
    expect(f.R.all_of<primitive::Cube>(viewport));
    snapshot::ApplyWorkspace(f.R, other_workspace);
    expect(!f.R.all_of<ViewCameraState>(viewport));
    expect(snapshot::SerializeWorkspace(f.R) == other_workspace);
    f.Audit();
}
// Exercise the production generic editors through ImGui activation and the history restage/commit path.
void TestHitEditor() {
    ScratchFixture f{"/tmp/mesheditor-scratch/history-hit-editor"};
    const auto object = f.AddEmpty("Object choice");
    f.AddEmpty("Other");
    const auto node = f.Do(action::selection::Pick{false, ObjectHit{{object}}});
    const auto hit = [&]() -> std::optional<SelectionHit> & {
        return std::get<action::selection::Pick>(std::get<action::selection::Action>(f.P->DraftOf(node).RecordedActions.front())).Hit;
    };
    auto *context = ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.IniFilename = nullptr;
    io.DisplaySize = {1000, 1000};
    io.Fonts->Build();
    HistoryPanel panel{{"History"}};
    const auto frame = [&](ImGuiID activate = 0) {
        if (activate) ImGui::ActivateItemByID(activate);
        ImGui::NewFrame();
        ImGui::SetNextWindowSize({900, 900});
        project::DrawHistoryPanel(*f.P, panel, true);
        ImGui::Render();
        f.P->Frame(action::Drain());
        f.Audit();
    };
    frame();
    const auto window_id = ImGui::FindWindowByName("History")->ID;
    const int index = 0;
    const auto action_id = ImHashData(&index, sizeof(index), ImHashData(&node, sizeof(node), window_id));
    const auto hit_id = ImHashStr("Hit", 0, action_id);
    const auto present_id = ImHashStr("##present", 0, hit_id);
    frame(present_id);
    expect(!hit());
    frame(present_id);
    expect(hit().has_value());
    expect(HitEntity(*hit()) == state::Null);
    expect(!f.R.all_of<Selected>(state::Document));
    ImGui::DestroyContext(context);
}
} // namespace

void TestHistory() {
    boost::ut::test("generic hit editors") = TestHitEditor;
    HistoryFixture f, other;
    for (const auto phase : {action::Phase::Record, action::Phase::Stage}) {
        Scenario(f, other, phase == action::Phase::Record ? "loop cut gesture" : "staged loop cut gesture", [phase](HistoryFixture &f) {
            const auto cube = f.AddCube();
            f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
            f.Do(action::view::SetEditMode{Element::Face});
            f.Image();
            const auto hit = RunEditElementPick(f.R, FirstViewport(f.R), Element::Edge, {32, 32});
            expect(hit.has_value());
            const auto before = f.P->History.MaterializeLive();
            const auto nodes = f.P->History.Nodes.size();
            const auto emit = [&](uint32_t cuts) {
                std::vector<action::Action> actions;
                actions.push_back(action::MakeAction(action::view::SetEditMode{Element::Edge}));
                actions.push_back(action::MakeAction(action::selection::ApplyEditElementClick{false, hit}));
                actions.push_back(action::MakeAction(action::mesh::LoopCut{cuts}));
                action::Emit(std::move(actions), phase);
                std::vector<action::Action> ignored;
                ignored.push_back(action::MakeAction(action::view::SetEditMode{Element::Vertex}));
                ignored.push_back(action::MakeAction(action::object::Delete{}));
                action::Emit(std::move(ignored));
                f.P->Frame(action::Drain());
                f.Audit();
            };
            emit(1);
            expect(f.R.get<const EditMode>(state::Document).Value == Element::Edge);
            expect(GetMesh(f.R, cube).VertexCount() == 12u);
            if (phase == action::Phase::Stage) {
                emit(2);
                expect(GetMesh(f.R, cube).VertexCount() == 16u);
                expect(f.P->History.Nodes.size() == nodes);
                action::Cancel();
                f.P->Frame(action::Drain());
                expect(f.P->History.MaterializeLive() == before);
                emit(1);
                action::Commit();
                f.P->Frame(action::Drain());
            }
            expect(f.P->History.Nodes.size() == nodes + 1);
            expect(f.P->DraftOf(*f.P->History.Present).RecordedActions.size() == 3u);
            f.Step(true);
            f.Undo();
            expect(f.P->History.MaterializeLive() == before);
            f.Redo();
            expect(GetMesh(f.R, cube).VertexCount() == 12u);
        });
    }
    Scenario(f, other, "gesture failure and background completion", [](HistoryFixture &f) {
        const auto cube = f.AddCube();
        const auto before = f.P->History.MaterializeLive();
        const auto nodes = f.P->History.Nodes.size();
        std::vector<action::Action> failed;
        failed.push_back(action::MakeAction(action::UpdateOf<&Transform::P>(cube, vec3{1, 0, 0})));
        failed.push_back(action::MakeAction(action::mesh::Boolean{}));
        failed.push_back(action::MakeAction(action::object::Delete{}));
        action::Emit(std::move(failed));
        f.P->Frame(action::Drain());
        auto &errors = f.R.Resources.get<action::Errors>().Messages;
        expect(errors.size() == 1u);
        errors.clear();
        expect(f.P->History.MaterializeLive() == before);
        expect(f.P->History.Nodes.size() == nodes);
        std::vector<action::Action> actions;
        actions.push_back(action::MakeAction(action::UpdateOf<&Transform::P>(cube, vec3{1, 0, 0})));
        actions.push_back(action::MakeAction(action::UpdateOf<&Transform::P>(cube, vec3{2, 0, 0})));
        action::Emit(std::move(actions));
        f.P->Enqueue(action::MakeAction(action::UpdateOf<&Transform::P>(cube, vec3{3, 0, 0})));
        f.P->Frame(action::Drain());
        expect(f.R.get<const Transform>(cube).P == vec3{3, 0, 0});
        expect(f.P->History.Nodes.size() == nodes + 2);
        f.Audit();
        f.Step(true);
        f.Undo();
        expect(f.R.get<const Transform>(cube).P == vec3{2, 0, 0});
        expect(f.P->DraftOf(*f.P->History.Present).RecordedActions.size() == 2u);
        f.Redo();
    });
    for (const bool orthographic : {false, true}) {
        Scenario(f, other, orthographic ? "orthographic selection replay" : "perspective selection replay", [orthographic](HistoryFixture &f) {
            auto &r = f.R;
            const auto viewport = FirstViewport(r);
            if (orthographic) f.Do(action::view::SetViewCameraLens{viewport, Orthographic{{6.f, 6.f}, 1000.f, .1f}});
            const auto cube = f.AddCube();
            f.AddCube();
            f.Image();
            const auto first = PickObject(r, viewport, {32, 32});
            f.Do(action::selection::Pick{false, first});
            r.Resources.get<FrameState>().DisplayFramebufferScale = {2, 2};
            f.ValidateReplay();
            r.Resources.get<FrameState>().DisplayFramebufferScale = {1, 1};
            f.Image();
            const auto next = PickObject(r, viewport, {32, 32}, true);
            expect(first && next && HitEntity(*first) != HitEntity(*next));
            f.Do(action::selection::Pick{true, next});
            expect(PickObject(r, viewport, {32, 32}, true) == first);
            f.Do(action::selection::DeselectAll{});
            for (const bool through : {false, true}) {
                f.Do(action::UpdateOf<&ViewportDisplay::XRaySolid>(viewport, through));
                f.Do(action::selection::ApplyBoxSelect{{vec2{.45f}, vec2{.55f}}, false});
            }
            const auto box_node = *f.P->History.Present;
            const auto target = f.Camera(viewport).Active.Target;
            f.MoveCamera(viewport, {100, 50, 25});
            auto &draft = f.P->DraftOf(box_node);
            auto &box = std::get<action::selection::ApplyBoxSelect>(std::get<action::selection::Action>(draft.RecordedActions.front()));
            box.Box = {vec2{0}, vec2{1}};
            f.P->RequestRestage();
            f.P->Frame(action::Drain());
            expect(r.all_of<Selected>(cube));
            expect(f.Camera(viewport).Active.Target == vec3{100, 50, 25});
            f.P->CancelGesture();
            f.Do(action::view::SetViewCameraTarget{viewport, target});
            f.Do(action::selection::Select{cube});
            f.Do(action::view::SetInteractionMode{InteractionMode::Edit});
            f.Do(action::view::SetEditMode{Element::Vertex});
            f.Do(action::selection::ApplyBoxSelect{{vec2{0}, vec2{1}}, false});
            f.Do(action::selection::ApplyEditElementClick{false, RunEditElementPick(r, viewport, Element::Vertex, {32, 32})});
            // Every history node replays after workspace navigation and display changes.
            f.Do(action::view::SetViewportShading{viewport, ViewportShadingMode::Wireframe});
            f.MoveCamera(viewport, {100, 50, 25});
        });
    }
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
        expect(f.R.all_of<primitive::Cube>(viewport));
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
