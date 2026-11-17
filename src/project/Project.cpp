#include "project/Project.h"

#include "Compress.h"
#include "Profile.h"
#include "action/Errors.h"
#include "armature/ArmatureComponents.h"
#include "editor/AudioIntegration.h"
#include "gizmo/GizmoInteraction.h"
#include "gltf/SourceAssets.h"
#include "mesh/MeshComponents.h"
#include "mesh/MeshStores.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "project/Assets.h"
#include "render/GpuBufferOps.h"
#include "render/GpuBuffers.h"
#include "render/MaterialComponents.h"
#include "render/RenderChanges.h"
#include "render/Textures.h"
#include "scene/Entity.h"
#include "selection/SelectionComponents.h"
#include "state/Registry.h"
#include "viewport/FrameState.h"
#include "viewport/Viewport.h"

#include <zpp_bits.h>

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
} // namespace

Project &Session(state::Registry &r) { return *r.Resources.get<Project *>(); }

Project::Project(state::Registry &r) : Entities(r, History, snapshot::SnapshotTable()), Recorded(NewRecording()), R(r) {
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
    History.Callbacks = {
        .Replay = [this](store::History &source, std::span<const std::byte> bytes, std::span<const uint32_t> workspaces) {
            const MeshStore::HistoryScope source_history{R.Resources.get<MeshStore>(), source};
            auto replayed = RunEncoded(*this, source, bytes, workspaces);
            Settle(EventPass::Settle);
            FinishSourceStates(R);
            EndMeshStates(R);
            EndCookStep(R);
            return replayed; },
        .Reset = [this](const std::filesystem::path &dir) {
            R.Resources.get<Assets>().Directory = dir;
            auto workspace = Entities.Workspace.Pin();
            ClearDocument();
            (void)Entities.Workspace.Restore(workspace);
            Entities.Workspace.Trie.Release(workspace);
            ReconcileWorkspace(R); },
        .BeforeSave = [this] { RecordMeshStates(R); },
        .BeforeRestore = [this] {
            WaitForRender(R);
            CancelModalSolves(R);
            DiscardPostedSteps();
            if (!GestureBase) ClearInteraction();
            Entities.BeginRestore(); },
        .AfterTracks = [this] {
            const profile::CpuScope scope{"RestoreAfterTracks"};
            const auto removed = Entities.RemovedEntities();
            // Instances destroy before the mesh entities they reference, covering each removed instance and each instance placed in a removed mesh's range.
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
    for (auto &node : History.Nodes) node.Editable = EditableEncoded(History.ActionsOf(node));
    History.Evict(MemoryCap);
    if (lock) DirectoryLock = std::move(lock);
    SavedPath = saved_path;
    RestoredWorkspace = saved ? std::move(saved->Workspace) : std::vector<std::byte>{};
    ResizeRecorded(0);
    DiscardPostedSteps();
    ++Revision;
    return true;
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

void Project::Settle(EventPass pass) {
    ProcessComponentEvents(R, pass);
    History.SettleHashes();
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
    // An event pass commits chains without waiting for them, and the reset frees what they reference.
    R.Resources.get<const mtl::Context>().WaitIdle();
    R.Resources.get<GpuBuffers>().Ctx.ReclaimRetiredBuffers(true);
    DiscardPostedSteps();
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
    for (auto type : snapshot::TypesOf(snapshot::Role::Workspace)) snapshot::SnapshotTable()[type].ClearRetained(R);
    return left;
}
void Project::ReturnToLeft(const LeftDocument &left) {
    ClearDocument();
    (void)History.Restore(left.Pin);
    (void)Entities.Workspace.Restore(left.Workspace);
    ReconcileWorkspace(R);
    Settle(EventPass::Settle);
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
    std::vector<state::Entity> restored_meshes;
    for (const auto &[type, entity, event] : Entities.TakeChanges()) {
        // The name hooks track created and destroyed names, and an overwritten name escapes them.
        names_changed |= type == state::Type<Name>() && event == state::Event::Update;
        if (type == state::Type<Armature>()) {
            for (const auto [object, armature] : R.view<const ArmatureObject>().each())
                if (armature.Entity == entity) R.remove<ArmaturePoseState>(object);
        } else if (type == state::Type<ArmatureObject>()) R.remove<ArmaturePoseState>(entity);
        if ((type == state::Type<MeshHandle>() || type == state::Type<EvaluatedMesh>()) && DrawsMesh(R, entity) && !R.all_of<procedure::ProcedureNode>(entity)) restored_meshes.push_back(entity);
        sources_changed |= type == state::Type<gltf::SourceAssets>();
        manifest_changed |= type == state::Type<MaterializedTexture>();
    }
    if (names_changed) RebuildEntityNames(R);
    // Restored sources can contain other pixels at a manifest's slots, so every texture and the IBL decode again.
    if (sources_changed) {
        ReleaseImportedTextures(R);
        ResetImportedEnvironment(R);
        R.Resources.get<RenderChanges>().SceneWorld.emplace(state::Document);
    } else if (manifest_changed) {
        std::vector<MaterializedTexture> listed;
        for (const auto [_, texture] : R.view<const MaterializedTexture>().each()) listed.push_back(texture);
        ReleaseUnlistedTextures(R, listed);
    }
    if (sources_changed || manifest_changed) R.Resources.get<RenderChanges>().MaterializedTextures.emplace(state::Document);
    ReconcileRestoredMeshes(R, std::move(restored_meshes));
    auto &buffers = R.Resources.get<GpuBuffers>();
    if (!buffers.Materials.History()->Trie.TakeChanged().empty()) {
        R.Resources.get<RenderChanges>().Materials.emplace(state::Document);
        R.Resources.get<procedure::NetworkChanges>().Pages.emplace(state::Document);
    }
    if (!buffers.MorphWeightBuffer.Buffer.History()->Trie.TakeChanged().empty()) R.Resources.get<RenderChanges>().MorphWeights.emplace(state::Document);
    Settle(EventPass::Restore);
}
} // namespace project
