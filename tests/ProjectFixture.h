#pragma once

#include "Compress.h"
#include "MeshChecks.h"
#include "TestPaths.h"
#include "WorkspaceState.h"
#include "action/Build.h"
#include "action/Errors.h"
#include "editor/Engine.h"
#include "mesh/Mesh.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshCreate.h"
#include "mesh/MeshStore.h"
#include "procedure/Cook.h"
#include "render/Instance.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "snapshot/Workspace.h"
#include "viewport/InteractionComponents.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"

#include <boost/ut.hpp>

#include <map>
#include <print>
#include <set>

// Compares rendered images, which --no-render skips.
inline bool Render{true};

// Each domain's elements as their sorted corner positions, sorted.
using Corner = std::array<float, 3>;
struct Shape {
    std::vector<Corner> Vertices;
    std::vector<std::array<Corner, 2>> Edges;
    std::vector<std::vector<Corner>> Faces;
    bool operator==(const Shape &) const = default;
};
inline Shape Describe(const MeshStore &meshes, uint32_t id, const GeometrySelection &elements) {
    const Mesh mesh{meshes, id};
    const auto point = [&](he::VH v) {
        const auto p = mesh.GetPosition(v);
        return Corner{p.x, p.y, p.z};
    };
    Shape shape;
    for (const auto v : elements.Vertices) shape.Vertices.push_back(point(he::VH{v}));
    for (const auto e : elements.Edges) {
        const auto h = mesh.GetHalfedge(he::EH{e}, 0u);
        const auto from = point(mesh.GetFromVertex(h)), to = point(mesh.GetToVertex(h));
        shape.Edges.push_back({std::min(from, to), std::max(from, to)});
    }
    for (const auto f : elements.Faces) {
        auto &corners = shape.Faces.emplace_back();
        for (const auto v : mesh.fv_range(he::FH{f})) corners.push_back(point(v));
        std::ranges::sort(corners);
    }
    std::ranges::sort(shape.Vertices);
    std::ranges::sort(shape.Edges);
    std::ranges::sort(shape.Faces);
    return shape;
}
inline Shape Describe(const MeshStore &meshes, uint32_t id) { return Describe(meshes, id, AllHandles(Mesh{meshes, id})); }

// An engine whose scenarios each begin a project in a scratch directory, auditing the project after each action and commit.
struct Fixture : Engine {
    // Runs after each action, commit and navigation, told whether the present node replays from its actions.
    std::function<void(bool replays)> Step;
    Fixture() : Engine{true} {}
    explicit Fixture(const TestDir &dir) : Fixture{} { Begin(dir.Path); }
    void Begin(const std::filesystem::path &dir) {
        Step = {};
        R.Resources.get<action::Errors>().Messages.clear();
        boost::ut::expect(P->New(dir));
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
    }
    std::vector<std::byte> Workspace() { return workspace::Serialize(workspace::Capture(R, R.Resources.emplace<PanelsState>())); }
    void Audit() {
        std::string why;
        const bool valid = P->Audit(why);
        if (!valid) std::println("audit: {}", why);
        boost::ut::expect(valid);
        for (const auto &message : R.Resources.get<action::Errors>().Messages) std::println("error: {}", message);
        boost::ut::expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    // Replays the present node's actions from its parent's state, which must reach the node's state.
    void ValidateReplay() {
        const auto node = P->History.Present.value();
        const auto why = P->History.ValidateReplay();
        if (!why.empty()) std::println("replay of node {} ({}) differs in {}", node, P->History.Nodes[node].Label, why);
        boost::ut::expect(why.empty());
    }
    uint32_t Do(auto a) {
        const auto node = P->Do(action::MakeAction(std::move(a)));
        Audit();
        if (Step) Step(true);
        return node.value();
    }
    // Commits direct scene writes as one history node, which no action replays.
    void Commit(std::string label) {
        P->Settle();
        P->Commit(std::move(label));
        Audit();
        if (Step) Step(false);
    }
    void Undo() {
        P->Undo();
        if (Step) Step(true);
    }
    void Redo() {
        P->Redo();
        if (Step) Step(true);
    }
    void Navigate(uint32_t node) {
        P->Navigate(node);
        if (Step) Step(true);
    }
    state::Entity Add(auto a) {
        Do(std::move(a));
        return FindActiveEntity(R);
    }
    state::Entity AddCube() { return Add(action::object::AddMeshPrimitive{primitive::Cube{}, std::make_unique<MeshInstanceCreateInfo>()}); }
    state::Entity AddEmpty(std::string name) {
        return Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Name = std::move(name)})});
    }
    MeshStore &Store() { return R.Resources.get<MeshStore>(); }
    std::vector<std::byte> Image() {
        // An opened document's viewport takes the fixture's extent.
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
        SubmitViewports(R);
        WaitForRender(R);
        const auto &image = R.get<const Viewport>(FirstViewport(R)).Targets.Screen->FinalColorImage;
        return ReadbackImageRgba8(R.Resources.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};

inline uint64_t Cooks(const state::Registry &r) { return r.Resources.get<const CookCount>().Value; }

// The live application at a node: the first viewport's image, the document, the workspace, and the record each mesh, owned mesh and Source node holds.
// The document and the drawn records belong to the node, while the image and the workspace follow the view, which history leaves alone, so only the present compares them.
struct Live {
    std::vector<std::byte> Image, Document, Workspace;
    std::map<state::Entity, Shape> Outputs;
    explicit Live(Fixture &f)
        : Image(Render ? f.Image() : std::vector<std::byte>{}), Document(f.P->History.MaterializeLive()), Workspace(snapshot::SerializeWorkspace(f.R)), Outputs(DrawnShapes(f)) {
        CheckRecords(f);
    }
    static std::map<state::Entity, Shape> DrawnShapes(Fixture &f) {
        std::map<state::Entity, Shape> shapes;
        for (const auto e : f.R.view<const EvaluatedMesh>()) shapes.emplace(e, Describe(f.Store(), *DrawnStoreId(f.R, e)));
        for (const auto e : f.R.view<const MeshHandle>()) shapes.emplace(e, Describe(f.Store(), *DrawnStoreId(f.R, e)));
        return shapes;
    }
    // Every drawn record's polygon tessellation agrees with its elements, and so do its render primitives once a frame has built them.
    // The records of the meshes in edit mode also keep a selection index that agrees with their elements.
    static void CheckRecords(Fixture &f) {
        const auto &meshes = f.Store();
        std::set<uint32_t> drawn;
        for (const auto e : f.R.view<const EvaluatedMesh>()) drawn.insert(*DrawnStoreId(f.R, e));
        for (const auto e : f.R.view<const MeshHandle>()) drawn.insert(*DrawnStoreId(f.R, e));
        for (const auto id : drawn) {
            const Mesh mesh{meshes, id};
            test::ExpectPolygonTessellation(meshes, mesh);
            if (Render && meshes.Get(id).RenderTopologies) test::ExpectRenderedGeometry(meshes, mesh);
        }
        if (f.R.get<const Interaction>(state::Document).Mode != InteractionMode::Edit) return;
        for (const auto [e, instance] : f.R.view<const Selected, const Instance>().each()) {
            if (const auto id = DrawnStoreId(f.R, instance.Entity)) test::ExpectSelectionIndex(meshes, *id);
        }
    }
    void Check(Fixture &f, bool present) const {
        if (present && Render) boost::ut::expect(f.Image() == Image);
        boost::ut::expect(f.P->History.MaterializeLive() == Document);
        if (present) boost::ut::expect(snapshot::SerializeWorkspace(f.R) == Workspace);
        boost::ut::expect(DrawnShapes(f) == Outputs);
        CheckRecords(f);
        f.Audit();
    }
};

// Every recorded node restores without cooking, hot and then cold, and each one made by actions replays.
// A history made by actions alone replays whole and reopens from an actions archive, and the present reopens from a saved project, each with the workspace applied.
// The present's actions replay from its parent to the same document with the same number of cooks hot, after a cold open and after clearing the history.
inline void CheckHistory(Fixture &f, Fixture &other, const std::map<uint32_t, Live> &nodes, const std::set<uint32_t> &direct, const TestDir &dir) {
    using boost::ut::expect;
    auto &p = *f.P;
    const auto present = p.History.Present.value();
    for (const auto &[node, live] : nodes) {
        const auto cooks = Cooks(f.R);
        p.Navigate(node);
        expect(Cooks(f.R) == cooks);
        live.Check(f, node == present);
        if (!direct.contains(node)) f.ValidateReplay();
    }
    const auto cap = std::exchange(p.MemoryCap, 0);
    for (const auto &[node, live] : nodes) {
        p.Navigate(node);
        live.Check(f, node == present);
    }
    p.MemoryCap = cap;
    p.Navigate(present);
    const auto &live = nodes.at(present);
    if (direct.empty()) {
        expect(p.Replay());
        live.Check(f, true);
        const TestDir archives{(dir.Path.string() + "-archive").c_str()}, opened{(dir.Path.string() + "-opened").c_str()};
        const auto archive = archives.Path / "Scene.actions";
        expect(p.SaveArchive(archive, project::ArchiveForm::Actions, live.Workspace));
        expect(ReadArchiveMetadata(archive).value() == live.Workspace);
        expect(Decompress(archive, opened.Path));
        expect(other.P->Open(opened));
        expect(other.P->History.Present == present);
        for (const auto &[node, node_live] : nodes) {
            other.P->Navigate(node);
            expect(other.P->History.MaterializeLive() == node_live.Document);
            other.Audit();
        }
        snapshot::ApplyWorkspace(other.R, live.Workspace);
        for (const auto &[node, node_live] : nodes) {
            other.P->Navigate(node);
            node_live.Check(other, node == present);
        }
    }
    // The present's actions replay from its parent with the same number of cooks hot, after a cold open and after clearing the history, so a restore the session made in place stays in place.
    const auto parent = p.History.Nodes[present].Parent;
    const bool replays = parent && !direct.contains(present);
    const auto actions = p.History.ActionsOf(p.History.Nodes[present]) | std::ranges::to<std::vector>();
    const auto workspaces = p.History.WorkspacesOf(p.History.Nodes[present]) | std::ranges::to<std::vector>();
    uint64_t replay_cooks = 0u;
    if (replays) {
        p.Navigate(*parent);
        const auto cooks = Cooks(f.R);
        expect(p.History.Callbacks.Replay(p.History, actions, workspaces).has_value());
        replay_cooks = Cooks(f.R) - cooks;
        p.Navigate(present);
    }
    const TestDir saved{(dir.Path.string() + "-saved").c_str()};
    const auto bytes = f.Workspace();
    expect(p.SaveAs(saved, bytes));
    expect(p.Close());
    expect(other.P->Open(saved.Path / "working", saved.Path / "Saved.project"));
    expect(other.P->History.Present == present);
    expect(other.P->RestoredWorkspace == bytes);
    snapshot::ApplyWorkspace(other.R, live.Workspace);
    live.Check(other, true);
    if (!replays) return;
    for (const bool clear : {false, true}) {
        other.P->Navigate(*parent);
        if (clear) expect(other.P->ClearHistory());
        const auto cooks = Cooks(other.R);
        expect(other.P->History.Callbacks.Replay(other.P->History, actions, workspaces).has_value());
        expect(other.P->History.MaterializeLive() == live.Document);
        expect(Cooks(other.R) - cooks == replay_cooks);
        other.Audit();
    }
}

// Runs `script` on a new project in `f`, recording each node it reaches after the check a fixture runs at each step, then checks the history it built through `other`.
template<typename F> void Scenario(F &f, Fixture &other, std::string name, auto &&script) {
    boost::ut::test(name) = [&] {
        const TestDir dir{std::format("/tmp/mesheditor-scratch/{}", name).c_str()};
        f.Begin(dir.Path);
        std::map<uint32_t, Live> nodes;
        std::set<uint32_t> direct;
        uint32_t last = 0;
        f.Step = [&](bool replays) {
            if constexpr (requires { f.CheckStep(); }) f.CheckStep();
            last = f.P->History.Present.value();
            nodes.insert_or_assign(last, Live{f});
            // A direct commit leaves the store in a state no action reproduces, so the nodes after it do not replay either.
            if (!replays) direct.insert(last);
        };
        script(f);
        f.Step = {};
        // A script may end on a node it did not record, such as a reference's release, which the checks leave for the last node it did.
        f.P->Navigate(last);
        CheckHistory(f, other, nodes, direct, dir);
    };
}

// Runs the procedure scenarios.
void TestProcedures();
