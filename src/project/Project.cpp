#include "project/Project.h"
#include "SortUnique.h"

#include "Compress.h"
#include "PathSerialize.h"
#include "Profile.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/Keying.h"
#include "armature/Armature.h"
#include "editor/AudioIntegration.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/SourceAssets.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStores.h"
#include "metal/Dispatch.h"
#include "numeric/Serialize.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "project/Assets.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/GpuSceneState.h"
#include "render/MaterialComponents.h"
#include "render/RenderChanges.h"
#include "render/Textures.h"
#include "selection/SelectionComponents.h"
#include "snapshot/Workspace.h"
#include "viewport/FrameState.h"
#include "viewport/GizmoDrag.h"
#include "viewport/ViewCameraOps.h"
#include "viewport/Viewport.h"
#include "viewport/ViewportEvents.h"
#include "viewport/ViewportRenderGpu.h"

namespace project {
namespace {
struct SavedState {
    store::HistoryPosition Position;
    std::vector<std::byte> Workspace;
};

std::vector<std::byte> SaveMetadata(const store::History &history, std::span<const std::byte> workspace) {
    std::vector<std::byte> bytes;
    const auto &node = history.Nodes[*history.Present];
    zpp::bits::out{bytes}(history.Present, node.Stamps, node.Roots, workspace).or_throw();
    return bytes;
}

std::optional<SavedState> ReadSavedState(Project &project, const std::filesystem::path &path) {
    if (const auto bytes = ReadArchiveMetadata(path)) {
        zpp::bits::in in{*bytes};
        SavedState state;
        if (zpp::bits::success(in(state)) && in.position() == bytes->size() &&
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
// A node's recorded actions, with no bytes for a node that records none, whose state no recorded actions rebuild.
// The output archive takes a mutable reference, since zpp aggregate reflection mis-encodes a const aggregate.
std::vector<std::byte> Encode(std::vector<action::Action> &recorded_actions) {
    std::vector<std::byte> bytes;
    if (!recorded_actions.empty()) zpp::bits::out{bytes}(recorded_actions).or_throw();
    return bytes;
}
std::vector<action::Action> Decode(std::span<const std::byte> bytes) {
    std::vector<action::Action> recorded_actions;
    if (!bytes.empty()) zpp::bits::in{bytes}(recorded_actions).or_throw();
    return recorded_actions;
}
bool Editable(std::span<const action::Action> recorded_actions) {
    return std::ranges::any_of(recorded_actions, [](const auto &recorded_action) {
        return action::VisitLeaf(recorded_action, []<typename L>(const L &) { return !std::is_empty_v<L>; });
    });
}
std::filesystem::path *SourcePath(action::Action &a) {
    return action::VisitLeaf(a, []<typename A>(A &leaf) -> std::filesystem::path * {
        if constexpr (std::same_as<A, action::io::Load> || std::same_as<A, action::io::LoadGltf> || std::same_as<A, action::io::LoadRealImpact> || std::same_as<A, action::object::ImportMesh> || std::same_as<A, action::audio::AssignVertexSamples> || std::same_as<A, action::audio::ApplyModalModel>) return &leaf.Path;
        else return nullptr;
    });
}
std::expected<std::vector<std::string>, std::string> ReferencedAssets(const store::History &history, std::span<const uint32_t> nodes) {
    namespace fs = std::filesystem;
    const Assets assets{.Directory = history.Dir};
    std::set<std::string> found;
    for (const auto node : nodes) {
        for (auto &recorded : Decode(history.ActionsOf(history.Nodes[node]))) {
            const auto *path = SourcePath(recorded);
            if (!path || !path->native().starts_with("asset:/")) continue;
            const auto relative = fs::path{path->native().substr(7)}.lexically_normal();
            if (relative.empty() || relative.is_absolute() || *relative.begin() == "..") return std::unexpected{"Invalid asset reference: " + path->string()};
            std::error_code ec;
            if (!fs::is_regular_file(assets.Resolve(*path), ec)) return std::unexpected{"Cannot read referenced asset: " + path->string()};
            found.insert((fs::path{Assets::DirectoryName} / relative).string());
        }
    }
    return std::vector<std::string>{found.begin(), found.end()};
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
        } else {
            return std::string{state::LeafName<A>()};
        }
    });
}

std::expected<void, std::string> WriteActionsArchive(const store::History &history, const std::filesystem::path &path, std::span<const std::byte> workspace) {
    File::TemporaryDirectory export_dir{history.Dir.parent_path()};
    const auto written = history.WriteActions(export_dir.Path / store::History::ActionsLogName);
    if (!written) return std::unexpected{written.error()};
    const auto referenced = ReferencedAssets(history, *written);
    if (!referenced) return std::unexpected{referenced.error()};
    for (const auto &entry : *referenced) {
        std::error_code ec;
        const auto target = export_dir.Path / entry;
        std::filesystem::create_directories(target.parent_path(), ec);
        if (!ec) std::filesystem::copy_file(history.Dir / entry, target, ec);
        if (ec) return std::unexpected{"Cannot export asset: " + ec.message()};
    }
    if (!Compress(export_dir.Path, path, workspace)) return std::unexpected{"cannot write " + path.string()};
    return {};
}

Project &Session(state::Registry &r) { return *r.Resources.get<Project *>(); }

Project::Project(state::Registry &r) : Entities(r, History, snapshot::SnapshotTable()), R(r) {
    R.Resources.emplace<Project *>(this);
    R.Resources.emplace<Assets>();
}
Project::~Project() {
    Close();
    R.Resources.erase<Project *>();
    R.Resources.erase<Assets>();
}

void Project::TrackStores() {
    auto &meshes = R.Resources.get<MeshStore>();
    meshes.Track(History);

    R.Resources.get<GpuBuffers>().Materials.Track(History, "material.values");
    R.Resources.get<GpuBuffers>().MorphWeightBuffer.Track(History, "morph.weights");
    R.Resources.get<MaterialStore>().Track(History);
    History.Callbacks = {
        .Replay = [this](store::History &source, std::span<const std::byte> bytes, std::span<const uint32_t> workspaces) {
            // A replayed step reproduces persistent state and leaves the workspace as it was, reconciled with the replayed document.
            const MeshStore::HistoryScope source_history{R.Resources.get<MeshStore>(), source};
            auto replayed = RunRecorded(source, Decode(bytes), workspaces);
            Settle(EventPass::Settle);
            // The replayed step ends as its commit did.
            FinishSourceStates(R);
            EndMeshStates(R);
            EndCookStep(R);
            return replayed; },
        .Reset = [this](const std::filesystem::path &dir) {
            R.Resources.get<Assets>().Directory = dir;
            // A replay from the empty state rebuilds the document and leaves the workspace as it was.
            auto workspace = Entities.Workspace.Pin();
            ClearDocument();
            (void)Entities.Workspace.Restore(workspace);
            Entities.Workspace.Trie.Release(workspace);
            ReconcileWorkspace(R); },
        .BeforeSave = [this] { RecordMeshStates(R); },
        .BeforeRestore = [this] {
            WaitForRender(R);
            CancelModalSolves(R);
            Deferred.clear();
            // A gesture's own restores keep its interaction, and every other restore drops it.
            if (!GestureBase) ClearInteraction();
            Entities.BeginRestore(); },
        .AfterTracks = [this] {
            const profile::CpuScope scope{"RestoreAfterTracks"};
            const auto removed = Entities.RemovedEntities();
            // Destroy instances before their referenced mesh entities: each removed instance, and each instance placed in a removed mesh's range.
            const auto &render_instances = std::as_const(R).storage<RenderInstance>();
            const auto object_ids = R.Resources.get<const GpuBuffers>().Instances.ObjectIdBuffer.GetSpan<uint32_t>();
            for (const auto e : removed) {
                R.remove<RenderInstance>(e);
                const auto *models = R.try_get<const ModelsBuffer>(e);
                if (!models) continue;
                for (uint32_t slot = models->InstanceRange.Offset; slot < models->InstanceRange.Offset + models->InstanceCount; ++slot) {
                    const auto instance_entity = render_instances.entity_at(ObjectIndex(object_ids[slot]));
                    if (const auto *ri = instance_entity != state::Null ? R.try_get<const RenderInstance>(instance_entity) : nullptr; ri && ri->Entity == e) R.remove<RenderInstance>(instance_entity);
                }
                FreeInstanceRange(R, models->InstanceRange);
            }
            R.Resources.get<MeshStore>().FinishRestore();
            Entities.FinishRestore(removed); },
        .AfterRestore = [this]() -> std::expected<void, std::string> {
            const auto errors = R.Resources.get<action::Errors>().Messages.size();
            AfterRestore();
            const auto &messages = R.Resources.get<action::Errors>().Messages;
            if (messages.size() != errors) return std::unexpected{messages[errors]};
            return {}; },
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
    auto &directory = R.Resources.get<Assets>().Directory;
    const auto previous = std::exchange(directory, dir);
    WaitForRender(R);
    Settle(EventPass::Settle);
    FinishSourceStates(R);
    EndCookStep(R);
    const bool editable = Editable(RecordedActions);
    const bool begun = History.Begin(dir, Encode(RecordedActions), std::exchange(RecordedWorkspaces, {}));
    ResizeRecorded(0);
    if (begun) {
        EndMeshStates(R);
        RecordMeshStates(R);
        History.Nodes.front().Editable = editable;
        DirectoryLock = std::move(lock);
        SavedPath.clear();
        RestoredWorkspace.clear();
        ++Revision;
    } else {
        directory = previous;
    }
    return begun;
}
bool Project::New(const std::filesystem::path &dir, bool empty) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        action::Fail(R, "Cannot create project directory: " + ec.message());
        return false;
    }
    if (History.Present && !Save()) return false;
    ResizeRecorded(0);
    auto left = LeaveDocument();
    // The root records how its scene arises, so its state replays from the empty scene.
    if (empty) Record(action::MakeAction(action::io::LoadEmptyScene{}), EventPass::Settle);
    else Record(action::MakeAction(action::io::LoadDefaultScene{}), EventPass::Settle);
    const bool begun = Begin(dir);
    if (!begun) ReturnToLeft(left);
    History.Release(left.Pin);
    Entities.Workspace.Trie.Release(left.Workspace);
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
    auto &directory = R.Resources.get<Assets>().Directory;
    const auto previous = std::exchange(directory, dir);
    const auto errors = R.Resources.get<action::Errors>().Messages.size();
    // Another project loads onto the empty document.
    std::optional<LeftDocument> left;
    if (dir != History.Dir) left = LeaveDocument();
    if (!History.Open(dir, saved ? &saved->Position : nullptr)) {
        directory = previous;
        if (left) {
            ReturnToLeft(*left);
            History.Release(left->Pin);
            Entities.Workspace.Trie.Release(left->Workspace);
        }
        const auto error = History.TakeIntegrityError();
        if (R.Resources.get<action::Errors>().Messages.size() == errors && !error.empty()) action::Fail(R, error);
        return false;
    }
    if (left) {
        History.Release(left->Pin);
        Entities.Workspace.Trie.Release(left->Workspace);
    }
    ReleaseGesture();
    for (auto &node : History.Nodes) node.Editable = Editable(Decode(History.ActionsOf(node)));
    // An actions open replays and pins every node on the path to the present node.
    History.Evict(MemoryCap);
    if (lock) DirectoryLock = std::move(lock);
    SavedPath = saved_path;
    RestoredWorkspace = saved ? std::move(saved->Workspace) : std::vector<std::byte>{};
    ResizeRecorded(0);
    Deferred.clear();
    ++Revision;
    return true;
}
bool Project::Save() {
    if (!History.Present) return true;
    WaitForRender(R);
    FinishGesture(EventPass::Settle);
    // Save playback progress since the last edit.
    Record(action::MakeAction(action::timeline::SetFrame{R.get<const TimelinePlayback>(state::Document).CurrentFrame}), EventPass::Settle);
    Commit("Playback position");
    return History.Save();
}
bool Project::SaveArchive(const std::filesystem::path &path, ArchiveForm form, std::span<const std::byte> workspace) {
    if (!Save()) return false;
    if (form == ArchiveForm::Actions) {
        const auto written = WriteActionsArchive(History, path, workspace);
        if (!written) action::Fail(R, "Cannot save actions '" + path.string() + "': " + written.error() + ".");
        return bool(written);
    }
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
    if (!SavedPath.empty() && destination == source) return SaveArchive(SavedPath, ArchiveForm::Project, workspace);
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
    const auto previous_lock = std::move(DirectoryLock);
    DirectoryLock = std::move(lock);
    R.Resources.get<Assets>().Directory = History.Dir;
    SavedPath = directory / "Saved.project";
    if (unnamed) fs::remove_all(previous, ec);
    return true;
}
bool Project::RevertSaved() {
    if (SavedPath.empty()) return false;
    auto saved = ReadSavedState(*this, SavedPath);
    if (!saved) return false;
    const auto node = History.FindPosition(saved->Position);
    if (!node) {
        action::Fail(R, "Saved position is missing from project history.");
        return false;
    }
    if (!Save()) return false;
    Navigate(*node);
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
    // The Source nodes and node results refer to states of the nodes Clear drops.
    RecordMeshStates(R);
    if (!History.Clear(saved ? &saved->Position : nullptr)) return false;
    ResizeRecorded(0);
    ++Revision;
    return true;
}
bool Project::Close() {
    ReleaseGesture();
    ResizeRecorded(0);
    RecordMeshStates(R);
    const bool closed = History.Close();
    DirectoryLock = {};
    SavedPath.clear();
    RestoredWorkspace.clear();
    return closed;
}

void Project::Tick(const action::Action &a, EventPass pass) {
    std::visit([&](const auto &domain) { Apply(R, domain); }, a);
    Settle(pass);
}
void Project::ResizeRecorded(size_t count) {
    while (RecordedWorkspaces.size() > count) {
        History.ReleaseWorkspace(RecordedWorkspaces.back());
        RecordedWorkspaces.pop_back();
    }
    RecordedActions.resize(count);
}
std::expected<void, std::string> Project::RunRecorded(store::History &source, std::span<const action::Action> actions, std::span<const uint32_t> workspaces) {
    if (actions.size() != workspaces.size()) return std::unexpected{"Action workspace count mismatch"};
    auto &frame = R.Resources.get<FrameState>();
    const auto saved = frame;
    auto live = Entities.Workspace.Pin();
    const auto errors = R.Resources.get<action::Errors>().Messages.size();
    for (size_t i = 0; i < actions.size(); ++i) {
        if (auto restored = source.RestoreWorkspace(workspaces[i]); !restored) {
            action::Fail(R, restored.error());
            break;
        }
        ReconcileWorkspace(R);
        const auto inputs = R.get<const ActionInputs>(state::Document);
        frame.DisplayFramebufferScale = inputs.FramebufferScale;
        if (R.get<const TimelinePlayback>(state::Document).CurrentFrame != inputs.CurrentFrame) {
            R.patch<TimelinePlayback>(state::Document, [&](auto &p) { p.CurrentFrame = inputs.CurrentFrame; });
            R.edit<PlaybackFrame>(state::Document).Value = float(inputs.CurrentFrame);
            Settle(EventPass::Settle);
        }
        R.edit<PlaybackFrame>(state::Document).Value = inputs.PlaybackFrame;
        if (frame.Scene != inputs.Scene) {
            frame.Scene = inputs.Scene;
            Settle(EventPass::Settle);
        }
        frame.DeltaTime = inputs.DeltaTime;
        frame.FixedFrameStep = inputs.FixedFrameStep;
        Tick(actions[i], inputs.Pass);
        if (R.Resources.get<action::Errors>().Messages.size() != errors) break;
    }
    EndGesture(EventPass::Settle);
    if (!Entities.Workspace.Restore(live)) action::Fail(R, "Live workspace restore mismatch");
    Entities.Workspace.Trie.Release(live);
    ReconcileWorkspace(R);
    frame = saved;
    const auto &messages = R.Resources.get<action::Errors>().Messages;
    if (messages.size() != errors) return std::unexpected{messages[errors]};
    return {};
}
void Project::Settle(EventPass pass) {
    ProcessComponentEvents(R, pass);
    History.SettleHashes();
}
bool Project::Record(action::Action a, EventPass pass, bool staged) { return Record(std::span{&a, 1}, pass, staged); }
bool Project::Record(std::span<action::Action> actions, EventPass pass, bool staged) {
    auto recordable = actions | std::views::filter([](const auto &a) { return action::IsRecordable(a); });
    if (staged && !GestureBase && !recordable.empty()) {
        GestureBase = History.Pin();
        GestureStart = RecordedActions.size();
    }
    const bool same_kind = staged && StageFirst && std::ranges::equal(recordable, std::span{RecordedActions}.subspan(*StageFirst), [](const auto &a, const auto &previous) { return Kind(a) == Kind(previous); });
    // Single-action previews can patch their existing nodes; a list replays from its stage base.
    const bool single = actions.size() == 1;
    if (same_kind && (!single || action::IsPreview(actions.front()))) {
        if (!single || !action::PatchesPreview(actions.front())) (void)History.Restore(*StageBase);
    } else if (staged && !recordable.empty()) {
        if (StageBase) History.Release(*StageBase);
        StageBase = History.Pin();
    }
    auto before = single ? std::optional<store::Snapshot>{} : History.Pin();
    const auto first = same_kind ? *StageFirst : RecordedActions.size();
    const auto errors = R.Resources.get<action::Errors>().Messages.size();
    bool recorded = false;
    for (auto &a : actions) {
        if (auto *path = SourcePath(a); path && !path->native().starts_with("asset:/")) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(*path, ec);
            if (ec) action::Fail(R, "Cannot resolve '" + path->string() + "': " + ec.message());
            else *path = R.Resources.get<Assets>().Reference(absolute);
        }
        const bool keep = action::IsRecordable(a);
        auto &frame = R.Resources.get<FrameState>();
        frame.Scene = ActingScene(R);
        const ActionInputs inputs{frame.Scene, frame.DeltaTime, R.get<const PlaybackFrame>(state::Document).Value, R.get<const TimelinePlayback>(state::Document).CurrentFrame, frame.FixedFrameStep, pass, frame.DisplayFramebufferScale};
        if (R.all_of<ActionInputs>(state::Document)) R.replace_if_changed(state::Document, inputs);
        else R.emplace<ActionInputs>(state::Document, inputs);
        const auto workspace = keep ? std::optional{History.CaptureWorkspace()} : std::nullopt;
        if (R.Resources.get<action::Errors>().Messages.size() == errors) Tick(a, pass);
        pass = EventPass::Settle;
        if (R.Resources.get<action::Errors>().Messages.size() != errors) {
            if (workspace) History.ReleaseWorkspace(*workspace);
            if (staged) CancelGesture();
            else if (before) {
                (void)History.Restore(*before);
                ResizeRecorded(first);
            }
            recorded = false;
            break;
        }
        if (!keep) continue;
        if (!recorded) {
            ResizeRecorded(first);
            if (staged) StageFirst = first;
        }
        RecordedActions.push_back(std::move(a));
        RecordedWorkspaces.push_back(*workspace);
        ++Revision;
        recorded = true;
    }
    if (before) History.Release(*before);
    return recorded;
}
std::optional<uint32_t> Project::Do(action::Action a, std::string label) {
    WaitForRender(R);
    if (label.empty()) label = Label(a);
    FinishGesture(EventPass::Settle);
    if (!Record(std::move(a), EventPass::Frame)) return History.Present;
    RecordKeys();
    return Commit(std::move(label));
}
void Project::RecordKeys() {
    if (!R.get<const Animations>(state::Document).Record) return;
    const auto seconds = animation::FrameSeconds(R, R.get<const TimelinePlayback>(state::Document).CurrentFrame);
    if (animation::AnyChanged(R, seconds)) Record(action::MakeAction(action::animation::RecordChanged{}), EventPass::Settle);
}
uint32_t Project::Commit(std::string label, std::optional<uint32_t> replace) {
    // The step's new Source nodes take the state it ends with, which the node it commits records.
    FinishSourceStates(R);
    EndCookStep(R);
    // A commit indexes the mesh records its actions wrote, so a later restore costs only its own change.
    R.Resources.get<MeshStore>().IndexHistory();
    const bool editable = Editable(RecordedActions);
    auto bytes = Encode(RecordedActions);
    const bool in_place = replace && History.Nodes[*replace].Children.empty();
    const auto created = uint32_t(History.Nodes.size());
    const auto node = in_place ? History.Replace(*replace, std::move(bytes), std::exchange(RecordedWorkspaces, {})) : History.Commit(std::move(label), std::move(bytes), std::exchange(RecordedWorkspaces, {}));
    // The states the step no longer refers to persist after the step's own content, which they share.
    EndMeshStates(R);
    // A commit matching an existing node's state keeps that node and its actions.
    if (in_place || node == created) History.Nodes[node].Editable = editable;
    Editing.reset();
    ResizeRecorded(0);
    History.Evict(MemoryCap);
    return node;
}
void Project::FinishGesture(EventPass pass) {
    if (!HasStaged()) return;
    // An edit keeps the node's label, and a new node names every distinct action in the gesture, in order.
    if (Editing) {
        Settle(pass);
        Commit(History.Nodes[*Editing].Label, Editing);
    } else {
        std::vector<std::string> names;
        for (size_t i = GestureStart; i < RecordedActions.size(); ++i) {
            if (auto name = Label(RecordedActions[i]); !std::ranges::contains(names, name)) names.push_back(std::move(name));
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
    R.remove<StartScreenTransform>(state::Document);
    Settle(pass);
}
Project::EditDraft &Project::DraftOf(uint32_t node) {
    if (!Draft || Draft->Node != node || Draft->Revision != History.Revision) Draft = EditDraft{node, History.Revision, Decode(History.ActionsOf(History.Nodes[node]))};
    return *Draft;
}
void Project::RestageDraft() {
    if (!Draft) return;
    if (Editing != Draft->Node && !EditNode(Draft->Node)) return;
    if (GestureBase) {
        if (!History.Restore(*GestureBase)) return;
    } else {
        GestureBase = History.Pin();
        GestureStart = RecordedActions.size();
    }
    ResizeRecorded(GestureStart);
    // Run a copy so the draft keeps its values for the next change.
    auto recorded_actions = Decode(Encode(Draft->RecordedActions));
    const auto workspaces = History.WorkspacesOf(History.Nodes[Draft->Node]);
    if (!RunRecorded(History, recorded_actions, workspaces)) {
        CancelGesture();
        return;
    }
    for (auto &recorded_action : recorded_actions) RecordedActions.push_back(std::move(recorded_action));
    for (auto id : workspaces) {
        History.RetainWorkspace(id);
        RecordedWorkspaces.push_back(id);
    }
    ++Revision;
}
void Project::ReleaseGesture() {
    EndCookStep(R);
    if (GestureBase) History.Release(*GestureBase);
    if (StageBase) History.Release(*StageBase);
    GestureBase.reset();
    StageBase.reset();
    StageFirst.reset();
}
void Project::ClearInteraction() {
    R.clear<StartTransform, StartBoneLength, StartPivot, StartScreenTransform, PendingTransform, action::DragFieldStart, AdditiveBoxSelectBaseline, PendingViewSetup>();
    if (auto *gizmo = R.try_edit<GizmoInteraction>(state::Document)) *gizmo = {};
    auto &frame = R.Resources.get<FrameState>();
    frame.BoxDrag.reset();
    frame.OperationStart.reset();
}
void Project::ClearDocument() {
    WaitForRender(R);
    Deferred.clear();
    ClearInteraction();
    R.clear<Viewport>();
    R.Reset();
    [[maybe_unused]] const auto document = R.create();
    assert(document == state::Document);
    SetupDocument(R);
}
Project::LeftDocument Project::LeaveDocument() {
    LeftDocument left{History.Pin(), Entities.Workspace.Pin()};
    ClearDocument();
    // The left document's retained values do not return to the next document.
    for (auto type : snapshot::TypesOf(snapshot::Role::Workspace)) snapshot::SnapshotTable()[type].ClearRetained(R);
    return left;
}
void Project::ReturnToLeft(const LeftDocument &left) {
    // The restore recreates every identity of the left document, and the applied workspace returns each one's values.
    ClearDocument();
    (void)History.Restore(left.Pin);
    (void)Entities.Workspace.Restore(left.Workspace);
    ReconcileWorkspace(R);
    Settle(EventPass::Settle);
}
void Project::CancelGesture() {
    if (!GestureBase) return;
    ResizeRecorded(0);
    ClearInteraction();
    (void)History.Restore(*GestureBase);
    ReleaseGesture();
    // A cancelled edit returns to the node it was replacing.
    if (const auto node = std::exchange(Editing, std::nullopt)) History.Navigate(*node);
    Draft.reset();
    ++Revision;
}
void Project::Frame(action::Drained drained) {
    WaitForRender(R);
    R.Resources.get<FrameState>().Scene = ActingScene(R);
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
        auto [actions, phase] = std::move(*drained.Emitted);
        if (phase == action::Phase::Cancel) {
            // A duplicate placement restarts under the new transform after its gesture is cancelled.
            std::optional<bool> duplicate;
            for (const auto &recorded_action : RecordedActions) {
                if (Is<action::object::Duplicate>(recorded_action)) duplicate = false;
                if (Is<action::object::DuplicateLinked>(recorded_action)) duplicate = true;
            }
            CancelGesture();
            if (duplicate) {
                Record(*duplicate ? action::MakeAction(action::object::DuplicateLinked{}) : action::MakeAction(action::object::Duplicate{}), EventPass::Settle, true);
            }
            for (const auto &a : actions) {
                Tick(a, pass);
                pass = EventPass::Settle;
            }
        } else {
            if (phase == action::Phase::Record) FinishGesture(EventPass::Settle);
            const auto label = Label(actions.back());
            const bool recordable = Record(actions, pass, phase == action::Phase::Stage);
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
void Project::Navigate(uint32_t node) {
    const profile::CpuScope scope{"Navigate"};
    CancelGesture();
    Editing.reset();
    ResizeRecorded(0);
    if (node == History.Present) History.Revert();
    else History.Navigate(node);
    History.Evict(MemoryCap);
    ++Revision;
}
bool Project::EditNode(uint32_t node) {
    const auto parent = History.Nodes[node].Parent;
    if (!parent) return false;
    Navigate(*parent);
    if (History.Present != parent) return false;
    Editing = node;
    return true;
}
bool Project::Replay() {
    FinishGesture(EventPass::Settle);
    ResizeRecorded(0);
    if (const auto diff = History.Replay(); !diff.empty()) {
        action::Fail(R, "Action replay differs in " + diff);
        return false;
    }
    History.Evict(MemoryCap);
    ++Revision;
    return true;
}
void Project::Undo() {
    CancelGesture();
    if (const auto node = History.UndoTarget()) Navigate(*node);
}
void Project::Redo() {
    CancelGesture();
    if (const auto node = History.RedoTarget()) Navigate(*node);
}
bool Project::Audit(std::string &why) {
    try {
        snapshot::VerifyCoverage(R);
    } catch (const std::exception &error) {
        why = error.what();
        return false;
    }
    // Resolve restored drawing entities by their store IDs.
    for (const auto [entity, evaluated] : R.view<const EvaluatedMesh>().each()) {
        if (MeshEntityOf(R, evaluated.StoreId) == entity) continue;
        why = "Mesh store " + std::to_string(evaluated.StoreId) + " does not map to its entity.";
        return false;
    }
    return R.Resources.get<const MeshStore>().Audit(why) && History.Audit(why);
}

void Project::AfterRestore() {
    const profile::CpuScope scope{"AfterRestore"};
    ReconcileWorkspace(R);
    const struct ReadOnly {
        state::Registry &R;
        explicit ReadOnly(state::Registry &r) : R(r) { R.DocumentReadOnly = true; }
        ~ReadOnly() { R.DocumentReadOnly = false; }
    } read_only{R};
    bool sources_changed = false, manifest_changed = false, names_changed = false;
    // Rebuild dirty roots for meshes whose handles or render records changed during restore.
    std::vector<state::Entity> restored_meshes;
    for (const auto &[type, entity, event] : Entities.TakeChanges()) {
        // The name hooks track created and destroyed names, and an overwritten name escapes them.
        names_changed |= type == state::Type<Name>() && event == state::Event::Update;
        if (type == state::Type<Armature>()) {
            for (const auto [object, armature] : R.view<const ArmatureObject>().each())
                if (armature.Entity == entity) R.remove<ArmaturePoseState>(object);
        } else if (type == state::Type<ArmatureObject>()) R.remove<ArmaturePoseState>(entity);
        if ((type == state::Type<MeshHandle>() || type == state::Type<EvaluatedMesh>()) && DrawsMesh(R, entity) && !R.all_of<procedure::NodeLayout>(entity)) restored_meshes.push_back(entity);
        sources_changed |= type == state::Type<gltf::SourceAssets>();
        manifest_changed |= type == state::Type<MaterializedTextures>();
    }
    if (names_changed) RebuildEntityNames(R);
    // Restored sources can contain other pixels at a manifest's slots, so every texture and the IBL decode again.
    // A manifest restored over unchanged sources keeps each texture it still lists, and the settle pass uploads the rest.
    if (sources_changed) {
        ReleaseImportedTextures(R);
        ResetImportedEnvironment(R);
        R.Resources.get<RenderChanges>().SceneWorld.emplace(state::Document);
    } else if (manifest_changed) {
        const auto *manifest = R.try_get<const MaterializedTextures>(state::Document);
        ReleaseUnlistedTextures(R, manifest ? std::span{manifest->Items} : std::span<const MaterializedTexture>{});
    }
    if (sources_changed || manifest_changed) R.Resources.get<RenderChanges>().MaterializedTextures.emplace(state::Document);
    ReconcileRestoredMeshes(R, std::move(restored_meshes));
    auto &buffers = R.Resources.get<GpuBuffers>();
    if (!buffers.Materials.History()->Trie.TakeChanged().empty()) R.Resources.get<RenderChanges>().Materials.emplace(state::Document);
    if (!buffers.MorphWeightBuffer.Buffer.History()->Trie.TakeChanged().empty()) R.Resources.get<RenderChanges>().MorphWeights.emplace(state::Document);
    Settle(EventPass::Restore);
}
} // namespace project
