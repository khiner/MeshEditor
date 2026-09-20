#include "project/Project.h"

#include "Compress.h"
#include "PathSerialize.h"
#include "ProcessEvents.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/AnimationData.h"
#include "animation/AnimationTimeline.h"
#include "animation/Keying.h"
#include "animation/MorphWeights.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "assets/ArchiveMesh.h"
#include "audio/RealImpact.h"
#include "editor/AudioIntegration.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/ArchiveSource.h"
#include "gltf/SourceAssets.h"
#include "gpu/PBRMaterial.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStore.h"
#include "numeric/Serialize.h"
#include "project/Assets.h"
#include "project/store/Pages.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/Instance.h"
#include "render/MaterialComponents.h"
#include "render/MeshBuffers.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/Selection.h"
#include "selection/SelectionComponents.h"
#include "state/Scene.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/InteractionComponents.h"
#include "viewport/RenderView.h"
#include "viewport/ViewCamera.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

#include <cstdio>

using state::Change;

namespace project {
namespace {
constexpr uint32_t SavedStateVersion = 2;
struct SavedState {
    store::HistoryPosition Position;
    std::vector<std::byte> Workspace;
};

std::vector<std::byte> SaveMetadata(const store::History &history, std::span<const std::byte> workspace) {
    std::vector<std::byte> bytes;
    const auto &node = history.Nodes[history.Present];
    zpp::bits::out{bytes}(SavedStateVersion, history.Present, node.Stamps, node.Roots, workspace).or_throw();
    return bytes;
}

std::optional<SavedState> ReadSavedState(Project &project, const std::filesystem::path &path) {
    if (const auto bytes = ReadArchiveMetadata(path)) {
        zpp::bits::in in{*bytes};
        uint32_t version{};
        SavedState state;
        if (zpp::bits::success(in(version, state)) && version == SavedStateVersion && in.position() == bytes->size() &&
            state.Position.Stamps.size() == project.History.Tracks.size() && state.Position.Roots.size() == project.History.Tracks.size()) return state;
    }
    action::Fail(project.R, "Cannot read saved project position from '" + path.string() + "'.");
    return std::nullopt;
}

auto Kind(const action::Action &a) {
    return std::pair{a.index(), std::visit([](const auto &domain) { return domain.index(); }, a)};
}
template<typename A> bool Is(const action::Action &a) {
    const auto *domain = std::get_if<action::DomainIndex<A>>(&a);
    return domain && std::holds_alternative<A>(*domain);
}
// A pixel-space action records the view it was made in, so it replays in that view's extent.
template<typename L>
concept ViewRecording = requires(L a) { { a.View } -> std::same_as<std::unique_ptr<RenderView> &>; };
bool RecordsView(const action::Action &a) {
    return action::VisitLeaf(a, []<typename L>(const L &) { return ViewRecording<L>; });
}

// A node's recorded actions, with no bytes for a baseline node that records none.
// The output archive takes a mutable reference, since zpp aggregate reflection mis-encodes a const aggregate.
std::vector<std::byte> Encode(std::vector<Project::RecordedAction> &recorded_actions) {
    std::vector<std::byte> bytes;
    if (!recorded_actions.empty()) zpp::bits::out{bytes}(recorded_actions).or_throw();
    return bytes;
}
std::vector<Project::RecordedAction> Decode(const std::vector<std::byte> &bytes) {
    std::vector<Project::RecordedAction> recorded_actions;
    if (!bytes.empty()) zpp::bits::in{bytes}(recorded_actions).or_throw();
    return recorded_actions;
}
} // namespace

std::string Label(const action::Action &a) {
    return action::VisitLeaf(a, []<typename A>(const A &leaf) {
        if constexpr (action::IsUpdate<A>) {
            const auto [component, field] = action::UpdatedField(leaf.ComponentType, leaf.Offset);
            return "Update " + component + ":" + field.Path;
        } else if constexpr (action::object::IsUpdateMaterial<A>) {
            return "Update Material:" + action::FieldAt<PBRMaterial>(leaf.Offset).Path;
        } else if constexpr (action::IsPatchFields<A>) {
            return [&]<typename C, typename F, size_t N>(const action::PatchFields<C, F, N> &patch) {
                std::string label = "Patch " + std::string{state::LeafName<C>()} + ":";
                for (size_t i = 0; i < N; ++i) label += (i ? ", " : "") + action::FieldAt<C>(patch.Offsets[i]).Path;
                return label;
            }(leaf);
        } else return std::string{state::LeafName<A>()};
    });
}

Project &Session(state::Scene &r) { return *r.Context.get<Project *>(); }

Project::Project(state::Scene &r) : Entities(r, History, snapshot::SnapshotTable()), R(r) {
    R.Context.emplace<Project *>(this);
    R.Context.emplace<Assets>();
}
Project::~Project() {
    Close();
    R.Context.erase<Project *>();
    R.Context.erase<Assets>();
}

void Project::TrackStores(state::Entity viewport) {
    Viewport = viewport;
    auto &meshes = R.Context.get<MeshStore>();
    meshes.Track(History);
    R.Context.get<GpuBuffers>().Materials.Track(History, "material.values");
    R.Context.get<GpuBuffers>().MorphWeightBuffer.Track(History, "morph.weights");
    R.Context.get<MaterialStore>().Track(History);
    History.SchemaRevision = 9;
    History.Callbacks = {
        .Replay = [this](const std::vector<std::byte> &bytes) { RunRecorded(Decode(bytes)); },
        .BeforeRestore = [this] {
            WaitForRender(R);
            CancelModalSolves(R);
            Deferred.clear();
            ClearInteraction();
            Entities.BeginRestore(); },
        .AfterTracks = [this] {
            const auto removed = Entities.RemovedEntities();
            // Destroy instances before their referenced mesh entities.
            if (!removed.empty()) {
                for (const auto [e, instance] : R.view<const RenderInstance>().each()) {
                    if (R.EntityAt(state::Index(e)) != e || R.EntityAt(state::Index(instance.Entity)) != instance.Entity) R.remove<RenderInstance>(e);
                }
            }
            for (const auto e : removed) {
                if (auto *buffers = R.try_edit<MeshBuffers>(e)) ReleaseMeshBuffers(R, *buffers);
                if (const auto *models = R.try_get<ModelsBuffer>(e)) FreeInstanceRange(R, models->InstanceRange);
            }
            R.Context.get<MeshStore>().FinishRestore();
            Entities.FinishRestore(removed); },
        .AfterRestore = [this] { AfterRestore(); },
    };
}

bool Project::Begin(const std::filesystem::path &dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    File::DirectoryLock lock{dir};
    if (ec || !lock) {
        action::Fail(R, "Cannot create or exclusively open project '" + dir.string() + "'.");
        return false;
    }
    auto &directory = R.Context.get<Assets>().Directory;
    const auto previous = std::exchange(directory, dir);
    WaitForRender(R);
    Settle(EventPass::Settle);
    const bool begun = History.Begin(dir);
    if (begun) {
        DirectoryLock = std::move(lock);
        SavedPath.clear();
        RestoredWorkspace.clear();
        ++Revision;
    } else directory = previous;
    return begun;
}
bool Project::New(const std::filesystem::path &dir, bool empty) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        action::Fail(R, "Cannot create project directory: " + ec.message());
        return false;
    }
    if (History.Present >= 0 && !Save()) return false;
    auto previous = History.Pin();
    const auto camera = R.get<const ViewCamera>(Viewport);
    RecordedActions.clear();
    Deferred.clear();
    ClearInteraction();
    ClearScene(R, Viewport);
    ClearAudioScene(R);
    if (!empty) Tick(action::MakeAction(action::io::LoadDefaultScene{}), EventPass::Settle);
    const bool begun = Begin(dir);
    if (!begun) {
        History.Restore(previous);
        R.replace<ViewCamera>(Viewport, camera);
        Settle(EventPass::Settle);
    }
    History.Release(previous);
    return begun;
}
bool Project::Open(const std::filesystem::path &dir, const std::filesystem::path &saved_path) {
    if (dir == History.Dir && saved_path.empty()) return true;
    std::optional<SavedState> saved;
    if (!saved_path.empty()) {
        saved = ReadSavedState(*this, saved_path);
        if (!saved) return false;
    }
    File::DirectoryLock lock;
    if (dir != History.Dir) lock = File::DirectoryLock{dir};
    if (dir != History.Dir && !lock) {
        action::Fail(R, "Project is already open or unavailable: '" + dir.string() + "'.");
        return false;
    }
    auto &directory = R.Context.get<Assets>().Directory;
    const auto previous = std::exchange(directory, dir);
    if (!History.Open(dir, saved ? &saved->Position : nullptr)) {
        directory = previous;
        return false;
    }
    ReleaseGesture();
    if (lock) DirectoryLock = std::move(lock);
    SavedPath = saved_path;
    RestoredWorkspace = saved ? std::move(saved->Workspace) : std::vector<std::byte>{};
    RecordedActions.clear();
    Deferred.clear();
    ++Revision;
    return true;
}
bool Project::Save() {
    if (History.Present < 0) return true;
    WaitForRender(R);
    FinishGesture(EventPass::Settle);
    // Save playback progress since the last edit.
    Record(action::MakeAction(action::timeline::SetFrame{R.get<const TimelinePlayback>(Viewport).CurrentFrame}), EventPass::Settle);
    Commit("Playback position");
    return History.Save();
}
bool Project::SaveArchive(const std::filesystem::path &path, std::span<const std::byte> workspace) {
    if (!Save()) return false;
    if (Compress(History.Dir, path, SaveMetadata(History, workspace))) return true;
    action::Fail(R, "Failed to save project '" + path.string() + "'.");
    return false;
}
bool Project::SaveAs(const std::filesystem::path &directory, std::span<const std::byte> workspace) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto destination = fs::weakly_canonical(directory, ec);
    if (ec) {
        action::Fail(R, "Cannot resolve project directory: " + ec.message());
        return false;
    }
    const auto source = fs::weakly_canonical(SavedPath.empty() ? History.Dir : SavedPath.parent_path(), ec);
    if (!SavedPath.empty() && destination == source) return SaveArchive(SavedPath, workspace);
    const auto relative = destination.lexically_relative(source), inverse = source.lexically_relative(destination);
    if (ec || relative.empty() || *relative.begin() != ".." || inverse.empty() || *inverse.begin() != "..") {
        action::Fail(R, "Choose a location outside the current project and its parent directories.");
        return false;
    }
    const bool exists = fs::exists(directory, ec);
    File::DirectoryLock destination_lock;
    if (exists) {
        const bool project = fs::is_regular_file(directory / "Saved.project", ec) && fs::is_directory(directory / "working", ec);
        if (!project && !fs::is_empty(directory, ec)) {
            action::Fail(R, "Cannot replace a nonempty folder that is not a MeshEditor project.");
            return false;
        }
        destination_lock = File::DirectoryLock{project ? directory / "working" : directory};
        if (!destination_lock) {
            action::Fail(R, "Project is already open or unavailable: '" + destination.string() + "'.");
            return false;
        }
    }
    if (!Save()) return false;
    File::TemporaryDirectory staging{directory.parent_path()};
    const auto fail = [&] {
        action::Fail(R, "Cannot save project directory '" + directory.string() + "'.");
        return false;
    };
    if (staging.Path.empty()) return fail();
    fs::copy(History.Dir, staging.Path / "working", fs::copy_options::recursive, ec);
    if (ec || !Compress(staging.Path / "working", staging.Path / "Saved.project", SaveMetadata(History, workspace))) return fail();
    File::DirectoryLock lock{staging.Path / "working"};
    if (!lock) return fail();
    const auto rename_flags = exists ? RENAME_SWAP : 0;
    if (::renamex_np(staging.Path.c_str(), directory.c_str(), rename_flags) != 0) return fail();
    const auto previous = History.Dir;
    const bool unnamed = SavedPath.empty();
    if (!History.Relocate(directory / "working")) {
        if (::renamex_np(directory.c_str(), staging.Path.c_str(), rename_flags) != 0 && exists) {
            action::Fail(R, "Previous project retained at '" + staging.Path.string() + "'.");
            staging.Path.clear();
        }
        return false;
    }
    auto previous_lock = std::move(DirectoryLock);
    DirectoryLock = std::move(lock);
    R.Context.get<Assets>().Directory = History.Dir;
    SavedPath = directory / "Saved.project";
    if (unnamed) fs::remove_all(previous, ec);
    return true;
}
bool Project::RevertSaved() {
    if (SavedPath.empty()) return false;
    auto saved = ReadSavedState(*this, SavedPath);
    if (!saved) return false;
    const int node = History.FindPosition(saved->Position);
    if (node < 0) {
        action::Fail(R, "Saved position is missing from project history.");
        return false;
    }
    if (!Save()) return false;
    Navigate(node);
    if (History.Present != node) return false;
    RestoredWorkspace = std::move(saved->Workspace);
    return true;
}
bool Project::ClearHistory() {
    WaitForRender(R);
    FinishGesture(EventPass::Settle);
    Settle(EventPass::Settle);
    std::optional<SavedState> saved;
    if (!SavedPath.empty()) {
        saved = ReadSavedState(*this, SavedPath);
        if (!saved) return false;
    }
    if (!History.Clear(saved ? &saved->Position : nullptr)) return false;
    RecordedActions.clear();
    ++Revision;
    return true;
}
bool Project::Close() {
    ReleaseGesture();
    const bool closed = History.Close();
    DirectoryLock = {};
    SavedPath.clear();
    RestoredWorkspace.clear();
    return closed;
}

void Project::Tick(const action::Action &a, EventPass pass) {
    std::visit([&](const auto &domain) { Apply(R, Viewport, domain); }, a);
    Settle(pass);
}
void Project::RunRecorded(std::span<const RecordedAction> recorded_actions) {
    auto &frame = R.Context.get<FrameState>();
    const auto saved = frame;
    const auto extent = R.Context.get<ViewportExtent>().Value;
    const auto live_view = static_cast<const CameraView &>(R.get<const ViewCamera>(Viewport));
    const auto set_view = [&](const CameraView &view) {
        if (static_cast<const CameraView &>(R.get<const ViewCamera>(Viewport)) == view) return;
        R.patch<ViewCamera>(Viewport, [&](auto &v) {
            static_cast<CameraView &>(v) = view;
            v.StopMoving();
        });
    };
    bool resized = false;
    for (const auto &[inputs, a] : recorded_actions) {
        // A pixel-space action renders its selection passes at the extent it recorded.
        if (RecordsView(a)) {
            R.Context.get<ViewportExtent>().Value = inputs.ViewportExtent;
            frame.DisplayFramebufferScale = inputs.DisplayFramebufferScale;
            resized |= inputs.ViewportExtent != extent || inputs.DisplayFramebufferScale != saved.DisplayFramebufferScale;
        }
        set_view(inputs.View);
        // Restore playback changes between recorded actions.
        if (R.get<const TimelinePlayback>(Viewport).CurrentFrame != inputs.CurrentFrame) {
            R.patch<TimelinePlayback>(Viewport, [&](auto &p) { p.CurrentFrame = inputs.CurrentFrame; });
            R.edit<PlaybackFrame>(Viewport).Value = float(inputs.CurrentFrame);
            Settle(EventPass::Settle);
        }
        R.edit<PlaybackFrame>(Viewport).Value = inputs.PlaybackFrame;
        frame.DeltaTime = inputs.DeltaTime;
        frame.FixedFrameStep = inputs.FixedFrameStep;
        Tick(a, inputs.Pass);
    }
    EndGesture(EventPass::Settle);
    set_view(live_view);
    frame = saved;
    R.Context.get<ViewportExtent>().Value = extent;
    // Resize the render targets back to the live extent before this frame draws.
    if (resized) Settle(EventPass::Settle);
}
void Project::Settle(EventPass pass) {
    ProcessComponentEvents(R, Viewport, pass);
    History.SettleHashes();
}
bool Project::Record(action::Action a, EventPass pass, bool staged) {
    auto *path = std::visit([](auto &domain) {
        return std::visit([]<typename A>(A &leaf) -> std::filesystem::path * {
            if constexpr (std::is_same_v<A, action::io::Load> || std::is_same_v<A, action::io::LoadGltf> || std::is_same_v<A, action::io::LoadRealImpact> || std::is_same_v<A, action::object::ImportMesh> || std::is_same_v<A, action::audio::AssignVertexSamples>) return &leaf.Path;
            else return nullptr;
        },
                          domain);
    },
                            a);
    if (path) {
        auto &assets = R.Context.get<Assets>();
        const auto ext = path->extension();
        const auto stored = Is<action::io::LoadRealImpact>(a) ? RealImpact::ArchiveSource(assets, *path) :
            ext == ".gltf" || ext == ".glb"                   ? gltf::ArchiveSource(assets, *path) :
            ext == ".obj"                                     ? ArchiveMesh(assets, *path) :
                                                                assets.Store(*path);
        if (!stored) {
            action::Fail(R, stored.error());
            return false;
        }
        *path = *stored;
    }
    const bool recordable = action::IsRecordable(a);
    if (staged && !GestureBase && recordable) {
        GestureBase = History.Pin();
        GestureStart = RecordedActions.size();
    }
    // A restarting operator reapplies from the gesture base.
    const bool same_kind = staged && StageFirst && Kind(RecordedActions[*StageFirst].Action) == Kind(a);
    if (same_kind && action::IsRestarting(a)) History.Restore(*GestureBase);
    const auto &frame = R.Context.get<const FrameState>();
    RecordedAction recorded_action{
        {R.get<const ViewCamera>(Viewport), R.Context.get<const ViewportExtent>().Value, frame.DisplayFramebufferScale, frame.DeltaTime,
         R.get<const PlaybackFrame>(Viewport).Value, R.get<const TimelinePlayback>(Viewport).CurrentFrame, frame.FixedFrameStep, pass},
        std::move(a),
    };
    Tick(recorded_action.Action, pass);
    if (!recordable) return false;
    ++Revision;
    // Only the latest update of a same-kind run is recorded.
    if (same_kind) RecordedActions.resize(*StageFirst);
    else if (staged) StageFirst = RecordedActions.size();
    RecordedActions.push_back(std::move(recorded_action));
    return true;
}
int Project::Do(action::Action a, std::string label) {
    WaitForRender(R);
    if (label.empty()) label = Label(a);
    FinishGesture(EventPass::Settle);
    if (!Record(std::move(a), EventPass::Frame)) return History.Present;
    RecordKeys();
    return Commit(std::move(label));
}
void Project::RecordKeys() {
    if (!R.get<const Animations>(Viewport).Record) return;
    const auto seconds = animation::FrameSeconds(R, Viewport, R.get<const TimelinePlayback>(Viewport).CurrentFrame);
    if (animation::AnyChanged(R, Viewport, seconds)) Record(action::MakeAction(action::animation::RecordChanged{}), EventPass::Settle);
}
int Project::Commit(std::string label, std::optional<int> replace) {
    auto bytes = Encode(RecordedActions);
    const bool in_place = replace && History.Nodes[*replace].Children.empty();
    const auto node = in_place ? History.Replace(*replace, std::move(bytes)) : History.Commit(std::move(label), std::move(bytes));
    Editing.reset();
    RecordedActions.clear();
    History.Evict(MemoryCap);
    return node;
}
void Project::FinishGesture(EventPass pass) {
    if (!HasStaged()) return;
    // An edit keeps the node's label, and a new node names every distinct action in the gesture, in order.
    if (Editing) {
        Commit(History.Nodes[*Editing].Label, Editing);
    } else {
        std::vector<std::string> names;
        for (size_t i = GestureStart; i < RecordedActions.size(); ++i) {
            if (auto name = Label(RecordedActions[i].Action); std::ranges::find(names, name) == names.end()) names.push_back(std::move(name));
        }
        std::string label;
        for (const auto &name : names) label += (label.empty() ? "" : ", ") + name;
        StageFirst.reset();
        EndGesture(pass);
        RecordKeys();
        Commit(label);
    }
    ReleaseGesture();
}
void Project::EndGesture(EventPass pass) {
    R.clear<StartTransform, StartBoneLength, StartPivot, action::DragFieldStart, AdditiveBoxSelectBaseline>();
    R.remove<StartScreenTransform>(Viewport);
    Settle(pass);
}
bool Project::Editable(int node) const {
    for (const auto &recorded_action : Decode(History.Nodes[node].Actions)) {
        if (action::VisitLeaf(recorded_action.Action, []<typename L>(const L &) { return !std::is_empty_v<L>; })) return true;
    }
    return false;
}
Project::EditDraft &Project::DraftOf(int node) {
    if (!Draft || Draft->Node != node || Draft->Revision != History.Revision) Draft = EditDraft{node, History.Revision, Decode(History.Nodes[node].Actions)};
    return *Draft;
}
void Project::RestageDraft() {
    if (!Draft) return;
    if (Editing != Draft->Node) EditNode(Draft->Node);
    if (GestureBase) History.Restore(*GestureBase);
    else {
        GestureBase = History.Pin();
        GestureStart = RecordedActions.size();
    }
    RecordedActions.resize(GestureStart);
    // Run a copy so the draft keeps its values for the next change.
    auto recorded_actions = Decode(Encode(Draft->RecordedActions));
    RunRecorded(recorded_actions);
    for (auto &recorded_action : recorded_actions) RecordedActions.push_back(std::move(recorded_action));
    ++Revision;
}
void Project::ReleaseGesture() {
    if (GestureBase) History.Release(*GestureBase);
    GestureBase.reset();
    StageFirst.reset();
}
void Project::ClearInteraction() {
    R.clear<StartTransform, StartBoneLength, StartPivot, StartScreenTransform, PendingTransform, action::DragFieldStart, AdditiveBoxSelectBaseline>();
    if (auto *gizmo = R.try_edit<GizmoInteraction>(Viewport)) *gizmo = {};
    auto &frame = R.Context.get<FrameState>();
    frame.BoxSelectStart.reset();
    frame.BoxSelectEnd.reset();
    frame.BoxSelectAdditive = false;
}
void Project::CancelGesture() {
    if (!GestureBase) return;
    RecordedActions.clear();
    History.Restore(*GestureBase);
    ReleaseGesture();
    // A cancelled edit returns to the node it was replacing.
    if (const auto node = std::exchange(Editing, std::nullopt)) History.Navigate(*node);
    Draft.reset();
    ++Revision;
}
void Project::Frame(action::Drained drained) {
    WaitForRender(R);
    if (const auto node = std::exchange(Navigation, {})) {
        Navigate(*node);
        return;
    }
    auto pass = EventPass::Frame;
    if (std::exchange(RestageRequested, false)) {
        RestageDraft();
        pass = EventPass::Settle;
    }
    if (drained.CancelRequested && HasStaged()) {
        CancelGesture();
        Settle();
        pass = EventPass::Settle;
    }
    if (drained.Emitted) {
        auto [a, phase] = std::move(*drained.Emitted);
        if (phase == action::Phase::Cancel) {
            // A duplicate placement restarts under the new transform after its gesture is cancelled.
            std::optional<bool> duplicate;
            for (const auto &recorded_action : RecordedActions) {
                if (Is<action::object::Duplicate>(recorded_action.Action)) duplicate = false;
                if (Is<action::object::DuplicateLinked>(recorded_action.Action)) duplicate = true;
            }
            CancelGesture();
            if (duplicate) {
                Record(*duplicate ? action::MakeAction(action::object::DuplicateLinked{}) : action::MakeAction(action::object::Duplicate{}), EventPass::Settle, true);
            }
            Tick(a);
        } else {
            if (phase == action::Phase::Record) FinishGesture(EventPass::Settle);
            const auto label = Label(a);
            const bool recordable = Record(std::move(a), pass, phase == action::Phase::Stage);
            if (phase == action::Phase::Record && recordable) {
                RecordKeys();
                Commit(label);
            }
        }
        pass = EventPass::Settle;
    }
    if (drained.CommitRequested && HasStaged()) {
        FinishGesture(pass);
        pass = EventPass::Settle;
    }
    for (auto &a : drained.System) Deferred.push_back(std::move(a));
    if (!HasStaged()) {
        auto deferred = std::exchange(Deferred, {});
        for (auto &a : deferred) {
            const auto label = Label(a);
            if (Record(std::move(a), pass)) Commit(label);
            pass = EventPass::Settle;
        }
    }
    if (pass == EventPass::Frame) Settle();
}
void Project::Navigate(int node) {
    CancelGesture();
    Editing.reset();
    RecordedActions.clear();
    if (node == History.Present) History.Revert();
    else History.Navigate(node);
    History.Evict(MemoryCap);
    ++Revision;
}
void Project::EditNode(int node) {
    Navigate(History.Nodes[node].Parent);
    Editing = node;
}
bool Project::Replay() {
    FinishGesture(EventPass::Settle);
    RecordedActions.clear();
    if (const auto diff = History.Replay(History.Present); !diff.empty()) {
        action::Fail(R, "Action replay differs in " + diff);
        return false;
    }
    History.Evict(MemoryCap);
    ++Revision;
    return true;
}
void Project::Undo() {
    CancelGesture();
    if (History.CanUndo()) Navigate(History.Nodes[History.Present].Parent);
}
void Project::Redo() {
    CancelGesture();
    if (History.CanRedo()) Navigate(History.Nodes[History.Present].Children.back());
}
bool Project::Audit(std::string &why) {
    try {
        snapshot::VerifyCoverage(R);
    } catch (const std::exception &error) {
        why = error.what();
        return false;
    }
    return History.Audit(why);
}

void Project::AfterRestore() {
    struct ReadOnly {
        state::Scene &R;
        explicit ReadOnly(state::Scene &r) : R(r) { R.DocumentReadOnly = true; }
        ~ReadOnly() { R.DocumentReadOnly = false; }
    } read_only{R};
    bool textures_changed = false, names_changed = false;
    for (const auto &[type, entity, event] : Entities.TakeChanges()) {
        names_changed |= type == state::Type<Name>();
        if (type == state::Type<Instance>() || type == state::Type<Hidden>()) {
            const auto *instance = R.try_get<const Instance>(entity);
            const bool visible = instance && !R.all_of<Hidden>(entity);
            if (const auto *render = R.try_get<const RenderInstance>(entity); render && (!visible || render->Entity != instance->Entity)) R.remove<RenderInstance>(entity);
            if (visible && !R.all_of<RenderInstance>(entity)) R.emplace<RenderInstance>(entity, instance->Entity, UINT32_MAX);
        }
        if (type == state::Type<Armature>()) R.remove<ArmaturePoseState>(entity);
        textures_changed |= type == state::Type<gltf::SourceAssets>() || type == state::Type<MaterializedTextures>();
    }
    if (names_changed) RebuildEntityNames(R);
    if (textures_changed) {
        ReleaseImportedTextures(R);
        ResetImportedEnvironment(R);
        reactive(R, Change::MaterializedTextures).emplace(Viewport);
        reactive(R, Change::SceneWorld).emplace(Viewport);
    }
    auto &meshes = R.Context.get<MeshStore>();
    const auto changes = meshes.TakeChanges();
    std::vector<Mesh> topology;
    std::vector<state::Entity> geometry;
    std::vector<MeshVertexChanges> positions;
    const auto editing = R.get<const Interaction>(Viewport).Mode == InteractionMode::Edit ? selection::ComputePrimaryEditInstances(R) : selection::PrimaryEditInstanceMap{};
    for (const auto [entity, handle] : R.view<const MeshHandle>().each()) {
        const auto it = std::ranges::lower_bound(changes, handle.StoreId, {}, &MeshStore::Change::StoreId);
        if (it == changes.end() || it->StoreId != handle.StoreId) continue;
        const bool sparse = (it->Bits & MeshStore::GeometryChanged) && !(it->Bits & ~(MeshStore::GeometryChanged | MeshStore::SelectionChanged)) && editing.contains(entity);
        if (it->Bits & (MeshStore::EntryChanged | MeshStore::TopologyChanged)) {
            topology.emplace_back(meshes, handle.StoreId);
            if (auto *buffer = R.try_edit<MeshBuffers>(entity)) ReleaseMeshBuffers(R, *buffer);
            R.remove<MeshBuffers>(entity);
            R.emplace<MeshBuffers>(entity, meshes.Arenas().Vertices.Slotted(meshes.Get(handle.StoreId).Vertices), SlottedRange{}, SlottedRange{}, SlottedRange{});
        } else if (sparse) {
            positions.push_back({entity, it->VertexRanges});
        } else if (it->Bits & (MeshStore::GeometryChanged | MeshStore::DeformChanged)) {
            geometry.push_back(entity);
        }
        if (it->Bits & MeshStore::ShadingChanged) reactive(R, Change::MeshShading).emplace(entity);
        if (it->Bits & MeshStore::SelectionChanged) R.Context.get<GpuSceneState>().EditSelectionDirty = true;
        if (!sparse && (it->Bits & ~MeshStore::SelectionChanged)) R.emplace_or_replace<MeshGeometryDirty>(entity, EditSelectionAfter::Keep);
    }
    meshes.RebuildDerived(topology);
    DeriveBaseNormalsNow(R, geometry);
    RefreshEditedPositions(R, positions);
    for (const auto &[entity, ranges] : positions) R.emplace_or_replace<MeshPositionsChanged>(entity);
    auto &buffers = R.Context.get<GpuBuffers>();
    if (!buffers.Materials.History()->Trie.TakeChanged().empty()) reactive(R, Change::Materials).emplace(Viewport);
    if (!buffers.MorphWeightBuffer.Buffer.History()->Trie.TakeChanged().empty()) reactive(R, Change::MorphWeights).emplace(Viewport);
    Settle(EventPass::Restore);
}
} // namespace project
