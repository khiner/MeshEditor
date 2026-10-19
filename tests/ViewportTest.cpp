#include "Paths.h"
#include "RunSuites.h"
#include "TestPaths.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "render/GpuSceneState.h"
#include "render/Textures.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"

#include <print>

using boost::ut::expect;

namespace {
constexpr uvec2 Extent{64, 64};

struct Fixture : Engine {
    explicit Fixture(const TestDir &dir) : Engine{true} {
        Viewports().All.front().State.Extent = Extent;
        expect(P->New(dir));
    }
    ::Viewports &Viewports() { return R.Context.get<::Viewports>(); }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::println("audit: {}", why);
        expect(valid);
        expect(R.Context.get<action::Errors>().Messages.empty());
    }
    uint32_t Do(auto a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        return node.value();
    }
    // Applies an unrecorded action, which leaves the history as it was.
    void DoLive(auto a) {
        const auto present = P->History.Present;
        const auto node_count = P->History.Nodes.size();
        expect(P->Do(action::MakeAction(std::move(a))) == present);
        expect(P->History.Nodes.size() == node_count);
        Audit();
    }
    // Adds a viewport sized to the fixture's extent.
    Viewport &AddViewport() {
        DoLive(action::view::AddViewport{});
        auto &viewport = Viewports().All.back();
        viewport.State.Extent = Extent;
        return viewport;
    }
    void Render() {
        SubmitViewports(R);
        WaitForRender(R);
    }
    std::vector<std::byte> Image(uint32_t id) {
        const auto &image = Viewports().Find(id)->Targets.Resources->FinalColorImage;
        return ReadbackImageRgba8(R.Context.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};

std::vector<uint32_t> Ids(const Viewports &viewports) {
    return viewports.All | std::views::transform(&Viewport::Id) | std::ranges::to<std::vector>();
}

// Two viewports with different shading render different images of one scene, each changed by unrecorded actions naming it alone, and removing one leaves the other rendering.
void TestTwoViewportsRender() {
    const TestDir dir{"/tmp/mesheditor-scratch/viewport-two-render"};
    Fixture f{dir};
    f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
    const auto document = f.P->History.MaterializeLive();
    expect(!action::IsRecordable(action::MakeAction(action::view::AddViewport{})));
    expect(!action::IsRecordable(action::MakeAction(action::view::RemoveViewport{1})));
    const auto solid_id = f.Viewports().All.front().Id;
    const auto wire_id = f.AddViewport().Id;
    expect(Ids(f.Viewports()) == std::vector{solid_id, wire_id});
    expect(wire_id == 2u);
    f.DoLive(action::view::SetViewportShading{solid_id, ViewportShadingMode::Solid});
    f.DoLive(action::view::SetViewportShading{wire_id, ViewportShadingMode::Wireframe});
    f.Render();
    const auto solid = f.Image(solid_id), wire = f.Image(wire_id);
    expect(!solid.empty());
    expect(solid.size() == wire.size());
    expect(solid != wire);

    f.Viewports().ActiveIndex = 1;
    f.DoLive(action::view::RemoveViewport{solid_id});
    expect(Ids(f.Viewports()) == std::vector{wire_id});
    expect(f.Viewports().ActiveIndex == 0u);
    // The service keeps its last viewport.
    f.DoLive(action::view::RemoveViewport{wire_id});
    expect(Ids(f.Viewports()) == std::vector{wire_id});

    f.DoLive(action::view::SetViewportShading{wire_id, ViewportShadingMode::Solid});
    f.Render();
    expect(f.Image(wire_id) == solid);
    const auto third_id = f.AddViewport().Id;
    expect(third_id == 3u);
    const auto wire_state = f.Viewports().Find(wire_id)->State;
    f.DoLive(action::UpdateOf<&ViewportDisplay::ShowOverlays>(action::OnViewport{third_id}, false));
    expect(!f.Viewports().Find(third_id)->State.Display.ShowOverlays);
    expect(f.Viewports().Find(wire_id)->State == wire_state);
    expect(f.P->History.MaterializeLive() == document);
    f.Audit();
}

// A workspace restores every viewport with its Id, visibility, and view values, matching the existing ones by Id.
void TestWorkspaceViewports() {
    const TestDir dir{"/tmp/mesheditor-scratch/viewport-workspace"}, restored_dir{"/tmp/mesheditor-scratch/viewport-workspace-restored"};
    Fixture f{dir};
    auto &first = f.Viewports().All.front();
    first.State.Camera.Active = ViewCamera{vec3{1, 2, 3}, vec3{0, 1, 0}, CameraLens{Perspective{.FieldOfViewRad = .7f, .FarClip = 100.f, .NearClip = .05f}}};
    first.State.Camera.LookThroughSaved = Defaults::ViewCamera;
    first.State.Environment.Name = "city";
    first.State.Lights.AmbientColor = {.1f, .2f, .3f};
    auto &second = f.AddViewport();
    second.Visible = false;
    second.State.Display.ViewportShading = ViewportShadingMode::Wireframe;
    second.State.Display.MotionBlur = MotionBlur{.Shutter = 1.f};
    second.State.PreviewLighting.Value.EnvIntensity = .25f;
    second.State.RenderLighting.Value.ExposureEV = 2.f;
    second.State.Orbit.Value = true;
    second.State.Extent = {32, 48};
    const auto bytes = workspace::Serialize(workspace::Capture(f.R, f.R.Context.get<const WindowsState>()));
    const auto &saved = f.Viewports().All;

    Fixture restored{restored_dir};
    // Ids 1 and 3 remain, so the state updates the first, creates Id 2, and drops Id 3.
    restored.AddViewport();
    restored.AddViewport();
    restored.DoLive(action::view::RemoveViewport{2});
    expect(Ids(restored.Viewports()) == std::vector{1u, 3u});
    const auto stored = workspace::Deserialize(bytes);
    expect(stored.has_value());
    if (!stored) return;
    workspace::Apply(restored.R, restored.R.Context.get<WindowsState>(), *stored);
    const auto &viewports = restored.Viewports().All;
    expect(Ids(restored.Viewports()) == std::vector{1u, 2u});
    if (viewports.size() != 2u) return;
    for (size_t i = 0; i < 2; ++i) {
        expect(viewports[i].Visible == saved[i].Visible);
        expect(viewports[i].State == saved[i].State);
    }
}

// A live action's view setup goes to the active displayed viewport in its next frame step, and navigating to a node recorded while one viewport looked through a camera leaves every viewport unchanged.
void TestViewSetupAndNavigation() {
    const TestDir dir{"/tmp/mesheditor-scratch/viewport-navigation"};
    Fixture f{dir};
    auto &p = *f.P;
    // The hidden active viewport hands the activity to the first visible one, which takes the load's setup.
    auto &second = f.AddViewport();
    second.Visible = false;
    f.Viewports().ActiveIndex = 1;
    const auto scene = std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "res/examples/DefaultScene/DefaultScene.gltf";
    const auto loaded = f.Do(action::io::LoadGltf{scene});
    expect(f.R.all_of<PendingViewSetup>(state::Document));
    f.Render();
    expect(f.Viewports().ActiveIndex == 0u);
    expect(!f.R.all_of<PendingViewSetup>(state::Document));
    const auto first_id = f.Viewports().All.front().Id;
    expect(f.Viewports().All.front().State.LookThroughEntity != state::Null);
    expect(second.State.LookThroughEntity == state::Null);
    // A presentation's setup changes the display, and the records follow the union that frame.
    f.Do(action::timeline::EnterPresentation{});
    expect(f.R.all_of<PendingViewSetup>(state::Document));
    f.Render();
    expect(!f.R.all_of<PendingViewSetup>(state::Document));
    expect(!WorkbenchShading(f.Viewports().All.front().State.Display.ViewportShading));
    expect(!f.Viewports().All.front().State.Display.ShowOverlays);
    expect(f.R.Context.get<const GpuSceneState>().Displays == VisibleDisplays(f.Viewports()));
    second.Visible = true;
    second.State.Display.ViewportShading = ViewportShadingMode::Wireframe;
    const auto second_id = second.Id;
    f.DoLive(action::view::OrbitViewCamera{second_id, {.3f, .1f}});
    const auto added = f.Do(action::object::AddMeshPrimitive{primitive::Cuboid{}, std::make_unique<MeshInstanceCreateInfo>()});
    f.DoLive(action::view::ExitLookThroughCamera{first_id});
    f.Render();
    const auto views = f.Viewports().All | std::views::transform(&Viewport::State) | std::ranges::to<std::vector>();
    const auto check = [&] {
        expect(!f.R.all_of<PendingViewSetup>(state::Document));
        f.Render();
        const auto &viewports = f.Viewports().All;
        expect(viewports.size() == views.size());
        for (size_t i = 0; i < std::min(viewports.size(), views.size()); ++i) expect(viewports[i].State == views[i]);
        f.Audit();
    };
    p.Navigate(loaded);
    check();
    expect(p.Replay());
    check();
    p.Navigate(added);
    check();
    // A live load's setup that no frame step took goes with the navigation.
    f.Do(action::io::LoadGltf{scene});
    expect(f.R.all_of<PendingViewSetup>(state::Document));
    p.Navigate(added);
    check();
}
} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Paths::Init(MESHEDITOR_BUILD_DIR, MESHEDITOR_BUILD_DIR);
    TestTwoViewportsRender();
    TestWorkspaceViewports();
    TestViewSetupAndNavigation();
    return RunSuites();
}
