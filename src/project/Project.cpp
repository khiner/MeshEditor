#include "project/Project.h"

#include "Compress.h"
#include "PathSerialize.h"
#include "ProcessEvents.h"
#include "action/Errors.h"
#include "animation/AnimationTimeline.h"
#include "animation/MorphWeightState.h"
#include "armature/Armature.h"
#include "armature/ArmatureComponents.h"
#include "assets/ArchiveMesh.h"
#include "audio/RealImpact.h"
#include "editor/AudioIntegration.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/ArchiveSource.h"
#include "gltf/SourceAssets.h"
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
std::string Label(const action::Action &a) {
    return std::visit([](const auto &domain) {
        return std::visit([]<typename A>(const A &) {
            const auto name = state::TypeName<A>();
            const auto base = name.substr(0, name.find('<'));
            return std::string{base.substr(base.rfind("::") + 2)};
        },
                          domain);
    },
                      a);
}
} // namespace
Project::Project(state::Scene &r) : Entities(r, History, snapshot::SnapshotTable()), R(r) {
    R.ctx().emplace<Project *>(this);
    R.ctx().emplace<Assets>();
}
Project::~Project() {
    Close();
    R.ctx().erase<Project *>();
    R.ctx().erase<Assets>();
}

void Project::TrackStores(state::Entity viewport) {
    Viewport = viewport;
    auto &meshes = R.ctx().get<MeshStore>();
    meshes.Track(History);
    R.ctx().get<GpuBuffers>().Materials.Track(History, "material.values");
    R.ctx().get<MaterialStore>().Track(History);
    History.SchemaRevision = 8;
    History.Callbacks = {
        .Replay = [this](const std::vector<std::byte> &bytes) {
            std::vector<Command> commands;
            zpp::bits::in{bytes}(commands).or_throw();
            auto &frame = R.ctx().get<FrameState>();
            const auto saved = frame;
            const auto extent = R.ctx().get<ViewportExtent>().Value;
            for (const auto &[inputs, a] : commands) {
                R.ctx().get<ViewportExtent>().Value = inputs.ViewportExtent;
                frame.DisplayFramebufferScale = inputs.DisplayFramebufferScale;
                // Replay commands with their recorded camera view.
                if (static_cast<const CameraView &>(R.get<const ViewCamera>(Viewport)) != inputs.View) {
                    R.patch<ViewCamera>(Viewport, [&](auto &v) {
                        static_cast<CameraView &>(v) = inputs.View;
                        v.StopMoving();
                    });
                }
                // Restore playback changes between recorded commands.
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
            R.clear<action::DragFieldStart>();
            frame = saved;
            R.ctx().get<ViewportExtent>().Value = extent; },
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
            R.ctx().get<MeshStore>().FinishRestore();
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
    auto &directory = R.ctx().get<Assets>().Directory;
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
    Commands.clear();
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
    auto &directory = R.ctx().get<Assets>().Directory;
    const auto previous = std::exchange(directory, dir);
    if (!History.Open(dir, saved ? &saved->Position : nullptr)) {
        directory = previous;
        return false;
    }
    ReleaseGesture();
    if (lock) DirectoryLock = std::move(lock);
    SavedPath = saved_path;
    RestoredWorkspace = saved ? std::move(saved->Workspace) : std::vector<std::byte>{};
    Commands.clear();
    Deferred.clear();
    ++Revision;
    return true;
}
bool Project::Save() {
    if (History.Present < 0) return true;
    WaitForRender(R);
    FinishGesture(EventPass::Settle);
    // Save playback progress since the last edit.
    ApplyCommand(action::MakeAction(action::timeline::SetFrame{R.get<const TimelinePlayback>(Viewport).CurrentFrame}), EventPass::Settle);
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
    R.ctx().get<Assets>().Directory = History.Dir;
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
    Commands.clear();
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
void Project::Settle(EventPass pass) {
    ProcessComponentEvents(R, Viewport, pass);
    History.SettleHashes();
}
bool Project::ApplyCommand(action::Action a, EventPass pass, bool staged) {
    auto *path = std::visit([](auto &domain) {
        return std::visit([]<typename A>(A &leaf) -> std::filesystem::path * {
            if constexpr (std::is_same_v<A, action::io::Load> || std::is_same_v<A, action::io::LoadGltf> || std::is_same_v<A, action::io::LoadRealImpact> || std::is_same_v<A, action::object::ImportMesh> || std::is_same_v<A, action::audio::AssignVertexSamples>) return &leaf.Path;
            else return nullptr;
        },
                          domain);
    },
                            a);
    if (path) {
        auto &assets = R.ctx().get<Assets>();
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
    if (staged && !GestureBase && recordable) GestureBase = History.Pin();
    const auto &frame = R.ctx().get<const FrameState>();
    Command command{
        {R.get<const ViewCamera>(Viewport), R.ctx().get<const ViewportExtent>().Value, frame.DisplayFramebufferScale, frame.DeltaTime,
         R.get<const PlaybackFrame>(Viewport).Value, R.get<const TimelinePlayback>(Viewport).CurrentFrame, frame.FixedFrameStep, pass},
        std::move(a),
    };
    Tick(command.Value, pass);
    if (!recordable) return false;
    ++Revision;
    // Replay requires the first gesture update for initialization and the latest for the final value.
    if (staged && StageFirst && Kind(Commands[*StageFirst].Value) == Kind(command.Value)) {
        if (Commands.size() > *StageFirst + 1) {
            Commands.back() = std::move(command);
            return true;
        }
    } else if (staged) StageFirst = Commands.size();
    Commands.push_back(std::move(command));
    return true;
}
int Project::Do(action::Action a, std::string label) {
    WaitForRender(R);
    if (label.empty()) label = Label(a);
    FinishGesture(EventPass::Settle);
    return ApplyCommand(std::move(a), EventPass::Frame) ? Commit(std::move(label)) : History.Present;
}
int Project::Commit(std::string label) {
    std::vector<std::byte> bytes;
    zpp::bits::out{bytes}(Commands).or_throw();
    const auto before = History.Present;
    const auto node = History.Commit(std::move(label), std::move(bytes));
    const bool baseline = !R.view<const StartTransform>().empty() || R.all_of<AdditiveBoxSelectBaseline>(Viewport);
    if (node != before || !baseline) Commands.clear();
    History.Evict(MemoryCap);
    return node;
}
void Project::FinishGesture(EventPass pass) {
    if (!HasStaged()) return;
    const auto label = Label(Commands[*StageFirst].Value);
    StageFirst.reset();
    ApplyCommand(action::MakeAction(action::view::EndGizmoDrag{}), pass);
    R.clear<action::DragFieldStart>();
    R.remove<AdditiveBoxSelectBaseline>(Viewport);
    Commit(label);
    Commands.clear();
    ReleaseGesture();
}
void Project::ReleaseGesture() {
    if (GestureBase) History.Release(*GestureBase);
    GestureBase.reset();
    StageFirst.reset();
}
void Project::ClearInteraction() {
    R.clear<StartTransform, StartBoneLength, StartScreenTransform, PendingTransform, action::DragFieldStart, AdditiveBoxSelectBaseline>();
    if (auto *gizmo = R.try_edit<GizmoInteraction>(Viewport)) *gizmo = {};
    auto &frame = R.ctx().get<FrameState>();
    frame.BoxSelectStart.reset();
    frame.BoxSelectEnd.reset();
    frame.BoxSelectStaged = false;
}
void Project::CancelGesture() {
    if (!GestureBase) return;
    Commands.clear();
    History.Restore(*GestureBase);
    ReleaseGesture();
    ++Revision;
}
void Project::Frame(action::Drained drained) {
    WaitForRender(R);
    if (const auto node = std::exchange(Navigation, {})) {
        Navigate(*node);
        return;
    }
    auto pass = EventPass::Frame;
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
            for (const auto &command : Commands) {
                if (Is<action::object::Duplicate>(command.Value)) duplicate = false;
                if (Is<action::object::DuplicateLinked>(command.Value)) duplicate = true;
            }
            CancelGesture();
            if (duplicate) {
                ApplyCommand(*duplicate ? action::MakeAction(action::object::DuplicateLinked{}) : action::MakeAction(action::object::Duplicate{}), EventPass::Settle, true);
            }
            Tick(a);
        } else if (HasStaged() && Is<action::view::EndGizmoDrag>(a)) {
            FinishGesture(pass);
        } else {
            if (phase == action::Phase::Record) FinishGesture(EventPass::Settle);
            const auto label = Label(a);
            const bool recordable = ApplyCommand(std::move(a), pass, phase == action::Phase::Stage);
            if (phase == action::Phase::Record && recordable) Commit(label);
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
            if (ApplyCommand(std::move(a), pass)) Commit(label);
            pass = EventPass::Settle;
        }
    }
    if (pass == EventPass::Frame) Settle();
}
void Project::Navigate(int node) {
    CancelGesture();
    Commands.clear();
    if (node == History.Present) History.Revert();
    else History.Navigate(node);
    History.Evict(MemoryCap);
    ++Revision;
}
bool Project::Replay() {
    FinishGesture(EventPass::Settle);
    Commands.clear();
    if (const auto diff = History.Replay(History.Present); !diff.empty()) {
        action::Fail(R, "Command replay differs in " + diff);
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
        if (type == state::Type<Armature>() || type == state::Type<ArmaturePose>()) R.remove<ArmaturePoseState>(entity);
        if (type == state::Type<MorphWeightState>()) R.remove<MorphWeightGpuRange>(entity);
        textures_changed |= type == state::Type<gltf::SourceAssets>() || type == state::Type<MaterializedTextures>();
    }
    if (names_changed) RebuildEntityNames(R);
    if (textures_changed) {
        ReleaseImportedTextures(R);
        ResetImportedEnvironment(R);
        reactive(R, Change::MaterializedTextures).emplace(Viewport);
        reactive(R, Change::SceneWorld).emplace(Viewport);
    }
    auto &meshes = R.ctx().get<MeshStore>();
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
        if (it->Bits & MeshStore::SelectionChanged) R.ctx().get<GpuSceneState>().EditSelectionDirty = true;
        if (!sparse && (it->Bits & ~MeshStore::SelectionChanged)) R.emplace_or_replace<MeshGeometryDirty>(entity, false);
    }
    meshes.RebuildDerived(topology);
    DeriveBaseNormalsNow(R, geometry);
    RefreshEditedPositions(R, Viewport, positions);
    for (const auto &[entity, ranges] : positions) R.emplace_or_replace<MeshPositionsChanged>(entity);
    auto &materials = R.ctx().get<GpuBuffers>().Materials;
    if (!materials.History()->Trie.TakeChanged().empty()) reactive(R, Change::Materials).emplace(Viewport);
    Settle(EventPass::Restore);
}
} // namespace project
