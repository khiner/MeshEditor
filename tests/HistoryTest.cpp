#include "ProjectFixture.h"
#include "audio/AudioOutputConfig.h"
#include "procedure/Macros.h"
#include "scene/HierarchyInstance.h"
#include "scene/SceneGraph.h"
#include "viewport/FrameState.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/ViewportScene.h"

#include <random>

using boost::ut::expect;

namespace {
struct HistoryFixture : Fixture {
    // The fixture's images size the first viewport's panel, so a first viewport's panel is left out under rendering.
    state::Entity AddViewport() {
        const auto before = R.view<const ViewportScene>() | std::ranges::to<std::vector>();
        Do(action::view::AddViewport{});
        const auto viewport = *std::ranges::find_if(R.view<const ViewportScene>(), [&](state::Entity e) { return !std::ranges::contains(before, e); });
        const bool sized = Render && viewport == FirstViewport(R);
        [&]<typename... Cs>(TypeList<Cs...>) { (expect((std::same_as<Cs, ViewportPanel> && sized) || R.get<const Cs>(viewport) == ViewDefault<Cs>()), ...); }(ViewValues{});
        return viewport;
    }
    void MoveCamera(state::Entity viewport, vec3 target) {
        Do(action::view::SetViewCameraTarget{viewport, target});
        expect(Camera(viewport) != ViewCameraState{});
    }
    ViewCameraState Camera(state::Entity viewport) const { return R.get<const ViewCameraState>(viewport); }
};

enum class WalkStep {
    AddCube,
    AddEmpty,
    AddViewport,
    RemoveViewport,
    ChangeView,
    SelectObject,
    ExtendActive,
    ParentToActive,
    MoveActive,
    Delete,
    Duplicate,
    LookThroughCamera,
    Undo,
    Redo,
    Navigate,
    Count,
};

// The rules a walk reaches at least once, each counted at a step that exercises it.
constexpr std::array WalkRules{"index reuse", "fresh identity", "kind change", "name collision", "loop in parents", "viewport removal", "look through"};

// Takes `steps` actions drawn uniformly from the walk vocabulary, checking after each action and navigation that:
// * An index freed along the present's path returns only at a higher generation.
// * Sibling names are unique, and an empty's taken name takes the suffix past every live one.
// * An identity carries only its kind's components, object or viewport.
// * A viewport added again after its add is undone takes the same identity and starts from defaults.
// * No node is its own ancestor, and parenting an object to its descendant reports the loop and leaves the hierarchy unchanged.
// * A removed viewport is invalid, every other viewport keeps its camera, and a view change records no node.
// * A viewport keeps looking through a camera whose creation is undone, and resolves it again on redo.
void RandomHistoryWalk(HistoryFixture &f, uint32_t seed, uint32_t steps) {
    auto &r = f.R;
    std::mt19937 rng{seed};
    const auto below = [&](size_t n) { return std::uniform_int_distribution<size_t>{0u, n - 1u}(rng); };
    const auto coordinate = [&] { return std::uniform_real_distribution<float>{-4.f, 4.f}(rng); };
    uint32_t step = 0;
    const auto at = [&] { return std::format("seed {} step {}", seed, step); };
    std::map<std::string_view, uint32_t> reached;
    // Per node, its live entities and the lowest generation each index freed along its path may return at, or zero.
    std::map<uint32_t, std::vector<state::Entity>> live;
    std::map<uint32_t, std::vector<uint32_t>> fresh_generation;
    std::map<state::Entity, bool> viewport_kind;
    const auto check = [&] {
        const auto node = f.P->History.Present.value();
        std::vector<state::Entity> alive;
        for (uint32_t i = 0; i < r.EntityCapacity(); ++i) {
            if (const auto e = r.EntityAt(i); e != state::Null) alive.push_back(e);
        }
        if (!fresh_generation.contains(node)) {
            std::vector<uint32_t> generations;
            if (const auto parent = f.P->History.Nodes[node].Parent) {
                generations = fresh_generation.at(*parent);
                for (const auto e : live.at(*parent)) {
                    if (r.valid(e)) continue;
                    generations.resize(std::max<size_t>(generations.size(), state::Index(e) + 1u));
                    generations[state::Index(e)] = std::max(generations[state::Index(e)], state::Generation(e) + 1u);
                }
            }
            fresh_generation.emplace(node, std::move(generations));
            live.emplace(node, alive);
        }
        const auto &generations = fresh_generation.at(node);
        for (const auto e : alive) {
            if (state::Index(e) < generations.size() && generations[state::Index(e)] != 0u) {
                expect(state::Generation(e) >= generations[state::Index(e)]) << at() << "index" << state::Index(e);
                ++reached["index reuse"];
            }
            const bool object = r.all_of<ObjectKind>(e), viewport = r.any_of<ViewportScene, ViewCameraState, Viewport>(e);
            expect(!(object && viewport)) << at() << "kinds of" << state::Index(e);
            if (!object && !viewport) continue;
            if (const auto [it, added] = viewport_kind.try_emplace(e, viewport); !added && it->second != viewport) {
                ++reached["kind change"];
                it->second = viewport;
            }
        }
        std::set<std::pair<state::Entity, std::string_view>> names;
        for (const auto [e, name] : r.view<const Name>().each()) expect(names.emplace(ParentOrNull(r, e), name.Value).second) << at() << "name" << name.Value;
        for (const auto e : r.view<const SceneParent>()) {
            uint32_t depth = 0;
            for (auto ancestor = e; ancestor != state::Null && depth <= r.EntityCapacity(); ancestor = ParentOrNull(r, ancestor)) ++depth;
            expect(depth <= r.EntityCapacity()) << at() << "parent cycle at" << state::Index(e);
        }
    };
    f.Step = [record = std::move(f.Step), &check](bool replays) {
        record(replays);
        check();
    };
    check();
    const auto pick = [&](auto &&view) {
        const auto entities = view | std::ranges::to<std::vector>();
        return entities.empty() ? state::Null : entities[below(entities.size())];
    };
    const auto objects = [&] { return r.view<const ObjectKind>(); };
    const auto viewports = [&] { return r.view<const ViewportScene>(); };
    const auto parents = [&] {
        std::map<state::Entity, state::Entity> found;
        for (const auto [e, parent] : r.view<const SceneParent>().each()) found.emplace(e, parent.Parent);
        return found;
    };
    // A request known to be a loop is rejected.
    const auto parent_to_active = [&](bool loop) {
        const auto before = parents();
        f.P->Do(action::MakeAction(action::object::ParentToActive{}));
        auto &messages = r.Resources.get<action::Errors>().Messages;
        if (messages.empty()) {
            expect(!loop) << at() << "a parent loop was accepted";
            f.Audit();
            f.Step(true);
            return;
        }
        expect(messages == std::vector<std::string>{"Loop in parents"}) << at();
        messages.clear();
        expect(parents() == before) << at() << "a rejected parent changed the hierarchy";
        ++reached["loop in parents"];
    };
    for (; step < steps; ++step) {
        switch (WalkStep(below(size_t(WalkStep::Count)))) {
            case WalkStep::AddCube: f.AddCube(); break;
            case WalkStep::AddEmpty: {
                const std::string requested = below(2) ? "Thing" : "Other";
                bool taken = false;
                uint64_t past = 0;
                for (const auto [e, name] : r.view<const Name>().each()) {
                    taken |= name.Value == requested;
                    if (name.Value.size() > requested.size() && NameStem(name.Value) == requested) past = std::max(past, std::stoull(name.Value.substr(requested.size() + 1u)));
                }
                const auto expected = taken ? std::format("{}_{}", requested, past + 1u) : requested;
                expect(r.get<const Name>(f.AddEmpty(requested)).Value == expected) << at() << expected;
                if (taken) ++reached["name collision"];
                break;
            }
            case WalkStep::AddViewport: {
                const auto viewport = f.AddViewport();
                if (below(2)) break;
                // The camera move records no node, so the undo removes the viewport.
                f.MoveCamera(viewport, {coordinate(), coordinate(), 1.f});
                f.Undo();
                expect(!r.valid(viewport) && f.AddViewport() == viewport) << at();
                ++reached["fresh identity"];
                break;
            }
            case WalkStep::RemoveViewport: {
                const auto viewport = pick(viewports());
                std::map<state::Entity, ViewCameraState> cameras;
                for (const auto e : viewports())
                    if (e != viewport) cameras.emplace(e, f.Camera(e));
                if (cameras.empty()) break;
                f.Do(action::view::RemoveViewport{viewport});
                expect(!r.valid(viewport)) << at();
                for (const auto &[e, camera] : cameras) expect(f.Camera(e) == camera) << at() << "camera of" << state::Index(e);
                ++reached["viewport removal"];
                break;
            }
            case WalkStep::ChangeView: {
                const auto viewport = pick(viewports());
                const auto node = f.P->History.Present;
                const auto nodes = f.P->History.Nodes.size();
                if (below(2)) {
                    f.MoveCamera(viewport, {coordinate(), coordinate(), 1.f});
                } else {
                    const auto lod_error = 1.f + float(below(4));
                    f.Do(action::UpdateOf<&ViewportDisplay::LodErrorPixels>(viewport, lod_error));
                    expect(r.get<const ViewportDisplay>(viewport).LodErrorPixels == lod_error) << at();
                }
                expect(f.P->History.Present == node && f.P->History.Nodes.size() == nodes) << at() << "a view change recorded a node";
                break;
            }
            case WalkStep::SelectObject:
                if (const auto e = pick(objects()); e != state::Null) f.Do(action::selection::Select{e});
                break;
            case WalkStep::ExtendActive:
                if (const auto e = pick(objects()); e != state::Null) f.Do(action::selection::ExtendActive{e});
                break;
            case WalkStep::ParentToActive: {
                bool loop = false;
                if (const auto child = pick(objects()), parent = pick(objects()); child != parent && below(2)) {
                    f.Do(action::selection::Select{child});
                    f.Do(action::selection::ExtendActive{parent});
                    parent_to_active(false);
                    if (ParentOrNull(r, child) == parent) {
                        f.Do(action::selection::Select{parent});
                        f.Do(action::selection::ExtendActive{child});
                        // A hierarchy member parents through its source, so only two plain objects make a known loop.
                        loop = !r.any_of<HierarchyMember>(child) && !r.any_of<HierarchyMember>(parent);
                    }
                }
                parent_to_active(loop);
                break;
            }
            case WalkStep::MoveActive:
                if (FindActiveEntity(r) != state::Null) f.Do(action::UpdateOf<&Transform::P>(action::OnActive{}, vec3{coordinate(), coordinate(), coordinate()}));
                break;
            case WalkStep::Delete: f.Do(action::object::Delete{}); break;
            case WalkStep::Duplicate:
                if (below(2)) f.Do(action::object::Duplicate{});
                else f.Do(action::object::DuplicateLinked{});
                break;
            case WalkStep::LookThroughCamera: {
                const auto viewport = pick(viewports());
                const CameraLens lens{Perspective{.FieldOfViewRad = .9f, .FarClip = 1000.f, .NearClip = .1f}};
                const auto camera = f.Add(action::object::AddCamera{std::make_unique<ObjectCreateInfo>(), lens});
                f.Do(action::view::SetLookThroughCamera{viewport, camera});
                expect(LookThroughCamera(r, viewport) == camera) << at();
                f.Undo();
                expect(LookThroughCamera(r, viewport) == state::Null && f.Camera(viewport).LookThrough == camera) << at();
                f.Redo();
                expect(LookThroughCamera(r, viewport) == camera) << at();
                ++reached["look through"];
                break;
            }
            case WalkStep::Undo: f.Undo(); break;
            case WalkStep::Redo: f.Redo(); break;
            case WalkStep::Navigate: f.Navigate(uint32_t(below(f.P->History.Nodes.size()))); break;
            case WalkStep::Count: break;
        }
    }
    for (const auto rule : WalkRules) expect(reached[rule] > 0u) << "seed" << seed << "never reached" << rule;
}

// Switching documents starts every workspace value from defaults and keeps the output device setting.
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
} // namespace

void TestHistory() {
    HistoryFixture f, other;
    for (const auto phase : {action::Phase::Record, action::Phase::Stage}) {
        Scenario(f, other, phase == action::Phase::Record ? "loop cut gesture" : "staged loop cut gesture", [phase](HistoryFixture &f) {
            const auto cube = f.AddCube();
            f.EnterEdit(Element::Face);
            f.Image();
            const auto hit = RunEditElementPick(f.R, FirstViewport(f.R), Element::Edge, {32, 32});
            expect(hit.has_value());
            const auto before = f.P->History.MaterializeLive();
            const auto nodes = f.P->History.Nodes.size();
            const auto emit = [&](uint32_t cuts) {
                std::vector<action::Action> actions;
                actions.push_back(action::MakeAction(action::view::SetEditMode{Element::Edge}));
                actions.push_back(action::MakeAction(action::selection::ApplyEditElementClick{false, hit}));
                actions.push_back(action::MakeAction(action::nodes::AppendMacro{procedure::LoopCutMacro, {{cuts, 0u, 0u, 0u}}}));
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
            f.EnterEdit(Element::Vertex);
            f.Do(action::selection::ApplyBoxSelect{{vec2{0}, vec2{1}}, false});
            f.Do(action::selection::ApplyEditElementClick{false, RunEditElementPick(r, viewport, Element::Vertex, {32, 32})});
            f.Do(action::view::SetViewportShading{viewport, ViewportShadingMode::Wireframe});
            f.MoveCamera(viewport, {100, 50, 25});
        });
    }
    // Removing a source's member removes each occurrence of it, through occurrences of occurrences.
    Scenario(f, other, "nested occurrence removal", [](HistoryFixture &f) {
        const auto parent = f.AddEmpty("Thing"), child = f.AddEmpty("Other");
        f.Do(action::selection::Select{child});
        f.Do(action::selection::ExtendActive{parent});
        f.Do(action::object::ParentToActive{});
        f.Do(action::selection::Select{parent});
        const auto inner = f.Add(action::object::DuplicateLinked{});
        const auto outer = f.Add(action::object::DuplicateLinked{});
        const auto members = f.R.get<const HierarchyInstance>(outer).Members;
        const auto occurrence = std::ranges::find_if(members, [&](const auto &member) { return member.Entity != outer; })->Entity;
        f.Do(action::selection::Select{occurrence});
        f.Do(action::object::Delete{});
        expect(!f.R.valid(child) && !f.R.valid(occurrence));
        expect(f.R.get<const HierarchyInstance>(inner).Members.size() == 1u && f.R.get<const HierarchyInstance>(outer).Members.size() == 1u);
    });
    // Seeds whose walks reach every rule, a reset-arena removal, a retallied shared record, a copy's name and a looked-through viewport's returning identity.
    for (const uint32_t seed : {15u, 16u, 27u, 78u}) {
        Scenario(f, other, std::format("random history walk {}", seed), [seed](HistoryFixture &f) { RandomHistoryWalk(f, seed, 300u); });
    }
    boost::ut::test("switching documents") = [&] { TestDocumentSwitch(f, other); };
}
