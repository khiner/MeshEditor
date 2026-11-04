#pragma once

#include "Compress.h"
#include "MeshChecks.h"
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
#include "procedure/Cook.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/SceneUpdates.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionGpu.h"
#include "snapshot/Workspace.h"
#include "viewport/InteractionComponents.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportComponents.h"

// Compares rendered images, which --no-render skips.
inline bool Render{true};

inline const auto RiggedSimple = std::filesystem::path{MESHEDITOR_SOURCE_DIR} / "external/glTF-Sample-Assets/Models/RiggedSimple/glTF/RiggedSimple.gltf";

using Corner = std::array<float, 3>;
// Each domain's elements as their sorted corner positions, sorted.
struct Shape {
    std::vector<Corner> Vertices;
    std::vector<std::array<Corner, 2>> Edges;
    std::vector<std::vector<Corner>> Faces;
    bool operator==(const Shape &) const = default;
};
inline Shape Describe(const MeshStore &meshes, uint32_t id, const GeometrySelection &elements) {
    const Mesh mesh{meshes, id};
    const auto point = [&](he::VH v) {
        const auto p = test::Position(mesh, v);
        return Corner{p.x, p.y, p.z};
    };
    Shape shape;
    for (const auto v : elements.Vertices) shape.Vertices.push_back(point(he::VH{v}));
    for (const auto e : elements.Edges) {
        const auto [a, b] = test::EdgeVertices(mesh, he::EH{e});
        const auto from = point(a), to = point(b);
        shape.Edges.push_back({std::min(from, to), std::max(from, to)});
    }
    for (const auto f : elements.Faces) {
        auto &corners = shape.Faces.emplace_back();
        for (const auto v : test::FaceVertices(mesh, he::FH{f})) corners.push_back(point(v));
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
    void Begin(const std::filesystem::path &dir) {
        Step = {};
        R.Resources.get<action::Errors>().Messages.clear();
        boost::ut::expect(P->New(dir));
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
    }
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
    void Stage(auto a) {
        action::Emit(std::move(a), action::Phase::Stage);
        P->Frame(action::Drain());
        boost::ut::expect(R.Resources.get<action::Errors>().Messages.empty());
    }
    void CommitGesture() {
        action::Commit();
        P->Frame(action::Drain());
        boost::ut::expect(!P->HasStaged());
        Audit();
        if (Step) Step(true);
    }
    void EnterEdit(Element element) {
        Do(action::view::SetInteractionMode{InteractionMode::Edit});
        Do(action::view::SetEditMode{.Mode = element});
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
    // Adds an object of the source at the world transform, selected alone, and returns its mesh entity and instance.
    std::pair<state::Entity, state::Entity> AddSourced(MeshSource source, const Transform &world = {}) {
        const auto added = ::AddMesh(R, CreateMesh(R, std::move(source)).StoreId, MeshInstanceCreateInfo{.Transform = world});
        Commit("Add mesh");
        return added;
    }
    state::Entity AddEmpty(std::string name) {
        return Add(action::object::AddEmpty{std::make_unique<ObjectCreateInfo>(ObjectCreateInfo{.Name = std::move(name)})});
    }
    MeshStore &Store() { return R.Resources.get<MeshStore>(); }
    std::vector<std::byte> Image() {
        R.replace<ViewportPanel>(FirstViewport(R), ViewportPanel{.Extent = {64, 64}});
        // A frame settles the events a restore or workspace apply left before it renders, as the app's loop does.
        P->Settle();
        SubmitViewports(R);
        WaitForRender(R);
        const auto &image = R.get<const Viewport>(FirstViewport(R)).Targets.Screen->FinalColorImage;
        return ReadbackImageRgba8(R.Resources.get<const mtl::Context>(), image, 0, 0, image.Extent);
    }
};

inline void SelectOrdinals(Fixture &f, state::Entity mesh, std::span<const uint32_t> ordinals, Element domain) {
    ApplyEditSelectionLists(f.R, std::array{std::pair{mesh, ordinals}}, domain);
    f.Commit("Select elements");
}

inline uint64_t Cooks(const state::Registry &r) { return r.Resources.get<const CookCount>().Value; }

// Recomputes each mesh's layout from its record, its instances and the interaction mode.
// The bounds runs share no storage, and the topology and sharp-edge tallies agree with a recount.
inline void CheckLayout(Fixture &f) {
    auto &r = f.R;
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const auto &meshes = r.Resources.get<const MeshStore>();
    const auto &render = meshes.Render();
    const bool edit = r.get<const Interaction>(state::Document).Mode == InteractionMode::Edit;
    const auto fail = [&](std::string_view what) {
        std::println("layout at node {}: {}", f.P->History.Present.value(), what);
        boost::ut::expect(false);
    };
    boost::ut::expect(r.view<const MeshLayoutDirty>().empty());
    std::array<std::vector<Range>, 3> runs;
    std::array<uint32_t, 3> topology_meshes{};
    uint32_t sharp_meshes = 0u;
    for (const auto [entity, models] : r.view<const ModelsBuffer>().each()) {
        const auto *record = TryRecordOf(r, entity);
        if (!record) continue;
        const auto states = buffers.Instances.States.front().GetSpan<uint8_t>({models.InstanceRange.Offset, models.InstanceCount});
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
        bool deformed = false;
        if (!edit) {
            for (const auto id : buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>({models.InstanceRange.Offset, models.InstanceCount})) {
                const auto instance = r.EntityAt(ObjectIndex(id));
                const auto *modifier = record->SkinBlocksReady ? r.try_get<const ArmatureModifier>(instance) : nullptr;
                const auto *pose_state = modifier ? r.try_get<const ArmaturePoseState>(modifier->ArmatureObjectEntity) : nullptr;
                deformed |= pose_state && modifier->SkinSlot < pose_state->GpuDeformRanges.size();
                deformed |= record->MorphBlocksReady && r.all_of<MorphWeightRange>(instance);
            }
        }
        const uint32_t count = deformed ? models.InstanceCount : 1u;
        if (run->Entries.Count != count) fail("entry count");
        if (run->Posed() != deformed || r.all_of<PosedLayout>(entity) != deformed) fail("posedness");
        const auto entries = buffers.BoundsReduceEntries.Get(run->Entries);
        for (uint32_t i = 0u; i < entries.size(); ++i) {
            if (entries[i].FirstInstance != models.InstanceRange.Offset + (deformed ? i : 0u) || entries[i].InstanceCount != (deformed ? 1u : models.InstanceCount)) fail("entry instances");
            if (entries[i].VertexOwner != record->Vertices.Index) fail("entry vertex owner");
        }
        runs[0].push_back(run->Entries);
        runs[1].push_back(run->DeriveEntries);
        runs[2].push_back(run->Tiles);
        if (!deformed) {
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

// Every instance has a render instance of its mesh in a slot of the mesh's instance range that holds its object ID.
// The slot's hidden bit in each shown scene agrees with the instance's Hidden component and scene membership.
// Each mesh's instance count equals its render instances, so every slot in a range belongs to a live instance.
inline void CheckInstances(Fixture &f) {
    const auto &r = f.R;
    const auto &buffers = r.Resources.get<const GpuBuffers>();
    const auto object_ids = buffers.Instances.ObjectIdBuffer.GetSpan<uint32_t>();
    const auto fail = [&](state::Entity e, std::string_view what) {
        std::println("instance {} at node {}: {}", state::Index(e), f.P->History.Present.value(), what);
        boost::ut::expect(false);
    };
    boost::ut::expect(r.view<const RenderInstance>().size() == r.view<const Instance>().size());
    std::map<state::Entity, uint32_t> placed;
    for (const auto [e, instance] : r.view<const Instance>().each()) {
        const auto *render = r.try_get<const RenderInstance>(e);
        const auto *models = render && render->Entity == instance.Entity ? r.try_get<const ModelsBuffer>(instance.Entity) : nullptr;
        if (!models || render->BufferIndex < models->InstanceRange.Offset || render->BufferIndex >= models->InstanceRange.Offset + models->InstanceCount || object_ids[render->BufferIndex] != ObjectId(e)) {
            fail(e, "not placed in its mesh's instance range");
            continue;
        }
        for (uint32_t i = 0u; i < buffers.Scenes.size(); ++i) {
            const bool hidden = (buffers.Instances.States[i].GetSpan<uint8_t>()[render->BufferIndex] & InstanceStateHidden) != 0u;
            if (hidden != HiddenIn(r, e, buffers.Scenes[i])) fail(e, "hidden bit");
        }
        ++placed[instance.Entity];
    }
    for (const auto [mesh, models] : r.view<const ModelsBuffer>().each()) {
        if (const auto it = placed.find(mesh); models.InstanceCount != (it != placed.end() ? it->second : 0u)) fail(mesh, "instance count");
    }
}

// The live application at a node: the first viewport's image, the document, the workspace, and the record each mesh, owned mesh and Source node holds.
// The document and the drawn records belong to the node, while the image and the workspace follow the view, which history leaves alone, so only the present compares them.
struct Live {
    std::vector<std::byte> Image, Document, Workspace;
    std::map<state::Entity, Shape> Outputs;
    explicit Live(Fixture &f)
        : Image(Render ? f.Image() : std::vector<std::byte>{}), Document(f.P->History.MaterializeLive()), Workspace(snapshot::SerializeWorkspace(f.R)), Outputs(DrawnShapes(f)) {
        CheckRecords(f);
    }
    static std::map<state::Entity, uint32_t> DrawnRecords(const Fixture &f) {
        std::map<state::Entity, uint32_t> records;
        for (const auto e : f.R.view<const EvaluatedMesh>()) records.emplace(e, *DrawnStoreId(f.R, e));
        for (const auto e : f.R.view<const MeshHandle>()) records.emplace(e, *DrawnStoreId(f.R, e));
        return records;
    }
    static std::map<state::Entity, Shape> DrawnShapes(Fixture &f) {
        std::map<state::Entity, Shape> shapes;
        for (const auto [e, id] : DrawnRecords(f)) shapes.emplace(e, Describe(f.Store(), id));
        return shapes;
    }
    // Every drawn record's polygon tessellation agrees with its elements, and so do its render primitives once a frame has built them.
    // The mesh layouts and the instance slots agree with a recount.
    // The records of the meshes in edit mode also keep a selection index that agrees with their elements.
    static void CheckRecords(Fixture &f) {
        const auto &meshes = f.Store();
        const auto drawn = DrawnRecords(f) | std::views::values | std::ranges::to<std::set>();
        for (const auto id : drawn) {
            const Mesh mesh{meshes, id};
            test::ExpectPolygonTessellation(meshes, mesh);
            if (Render && meshes.Get(id).RenderTopologies) test::ExpectRenderedGeometry(meshes, mesh);
        }
        CheckLayout(f);
        CheckInstances(f);
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
// The present's actions replay from its parent to the same document and cook count hot, after a cold open and after clearing the history.
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
        snapshot::ApplyWorkspace(other.R, live.Workspace);
        for (const auto &[node, node_live] : nodes) {
            other.P->Navigate(node);
            node_live.Check(other, node == present);
        }
    }
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
    const auto bytes = workspace::Serialize(workspace::Capture(f.R, f.R.Resources.emplace<PanelsState>()));
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
