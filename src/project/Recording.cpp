#include "project/Recording.h"

#include "Compress.h"
#include "PathSerialize.h"
#include "action/Dispatch.h"
#include "action/Errors.h"
#include "animation/Keying.h"
#include "mesh/MeshStore.h"
#include "numeric/Serialize.h"
#include "procedure/Cook.h"
#include "procedure/Procedure.h"
#include "project/Assets.h"
#include "viewport/FrameState.h"
#include "viewport/Viewport.h"

namespace {
// The frame's gesture, which is open while it has no actions.
std::optional<std::pair<std::vector<action::Action>, action::Phase>> Emitted;
// The posted actions, each applied and committed as its own history step.
std::vector<action::Action> Steps;
bool CommitRequested = false, CancelRequested = false;

void Push(action::Destination destination, action::Action a) {
    if (destination == action::Destination::Step) Steps.push_back(std::move(a));
    else if (Emitted) Emitted->first.push_back(std::move(a));
}
} // namespace

namespace action {
bool Begin(Phase phase) {
    if (Emitted && !Emitted->first.empty()) return false;
    Emitted.emplace(std::vector<Action>{}, phase);
    return true;
}
void Append(Destination d, Core a) { Push(d, std::move(a)); }
void Append(Destination d, selection::Action a) { Push(d, std::move(a)); }
void Append(Destination d, object::Action a) { Push(d, std::move(a)); }
void Append(Destination d, view::Action a) { Push(d, std::move(a)); }
void Append(Destination d, physics::Action a) { Push(d, std::move(a)); }
void Append(Destination d, audio::Action a) { Push(d, std::move(a)); }
void Append(Destination d, bone::Action a) { Push(d, std::move(a)); }
void Append(Destination d, timeline::Action a) { Push(d, std::move(a)); }
void Append(Destination d, io::Action a) { Push(d, std::move(a)); }
void Append(Destination d, animation::Action a) { Push(d, std::move(a)); }
void Append(Destination d, mesh::Action a) { Push(d, std::move(a)); }
void Append(Destination d, nodes::Action a) { Push(d, std::move(a)); }
void Append(Destination d, scene::Action a) { Push(d, std::move(a)); }
void Commit() { CommitRequested = true; }
void Cancel() { CancelRequested = true; }
} // namespace action

namespace project {
struct Recording {
    std::vector<action::Action> Actions; // The actions the open gesture has recorded
    std::optional<EditDraft> Draft;
};

namespace {
auto Kind(const action::Action &a) {
    return std::pair{a.index(), std::visit([](const auto &domain) { return domain.index(); }, a)};
}
template<typename A> bool Is(const action::Action &a) {
    const auto *domain = std::get_if<action::DomainIndex<A>>(&a);
    return domain && std::holds_alternative<A>(*domain);
}
bool ContinuesSelection(const action::Action &a) {
    if (const auto *view = std::get_if<action::view::Action>(&a)) {
        if (const auto *tool = std::get_if<action::view::SetActiveTool>(view)) return tool->Value == action::view::SetActiveTool::Tool::SelectBox || tool->Value == action::view::SetActiveTool::Tool::SelectClick;
    }
    return !action::IsRecordable(a) || Is<action::nodes::FinishSelection>(a) || Is<action::selection::ApplyEditElementClick>(a) ||
        Is<action::selection::ApplyBoxSelect>(a) || Is<action::selection::SelectAll>(a) || Is<action::selection::DeselectAll>(a) || Is<action::view::SetEditMode>(a);
}

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

void Tick(Project &session, const action::Action &a, EventPass pass = EventPass::Frame) {
    std::visit([&](const auto &domain) { Apply(session.R, domain); }, a);
    session.Settle(pass);
}
std::expected<void, std::string> RunRecorded(Project &session, store::History &source, std::span<const action::Action> actions, std::span<const uint32_t> workspaces) {
    if (actions.size() != workspaces.size()) return std::unexpected{"Action workspace count mismatch"};
    auto &frame = session.R.Resources.get<FrameState>();
    const auto saved = frame;
    auto live = session.Entities.Workspace.Pin();
    const auto errors = session.R.Resources.get<action::Errors>().Messages.size();
    for (size_t i = 0; i < actions.size(); ++i) {
        if (auto restored = source.RestoreWorkspace(workspaces[i]); !restored) {
            action::Fail(session.R, restored.error());
            break;
        }
        ReconcileWorkspace(session.R);
        const auto inputs = session.R.get<const ActionInputs>(state::Document);
        frame.DisplayFramebufferScale = inputs.FramebufferScale;
        if (session.R.get<const TimelinePlayback>(state::Document).CurrentFrame != inputs.CurrentFrame) {
            session.R.patch<TimelinePlayback>(state::Document, [&](auto &p) { p.CurrentFrame = inputs.CurrentFrame; });
            session.R.edit<PlaybackFrame>(state::Document).Value = float(inputs.CurrentFrame);
            session.Settle(EventPass::Settle);
        }
        session.R.edit<PlaybackFrame>(state::Document).Value = inputs.PlaybackFrame;
        if (frame.Scene != inputs.Scene) {
            frame.Scene = inputs.Scene;
            session.Settle(EventPass::Settle);
        }
        frame.DeltaTime = inputs.DeltaTime;
        frame.FixedFrameStep = inputs.FixedFrameStep;
        frame.GestureOpen = inputs.Staged;
        Tick(session, actions[i], inputs.Pass);
        if (session.R.Resources.get<action::Errors>().Messages.size() != errors) break;
    }
    session.EndGesture(EventPass::Settle);
    if (!session.Entities.Workspace.Restore(live)) action::Fail(session.R, "Live workspace restore mismatch");
    session.Entities.Workspace.Trie.Release(live);
    ReconcileWorkspace(session.R);
    frame = saved;
    const auto &messages = session.R.Resources.get<action::Errors>().Messages;
    if (messages.size() != errors) return std::unexpected{messages[errors]};
    return {};
}
// Captures the workspace before each recorded action, then applies it.
bool Record(Project &session, std::span<action::Action> actions, EventPass pass, bool staged = false) {
    auto recordable = actions | std::views::filter([](const auto &a) { return action::IsRecordable(a); });
    if (staged && !session.GestureBase && !recordable.empty()) {
        session.GestureBase = session.History.Pin();
        session.GestureStart = session.Recorded->Actions.size();
    }
    const bool same_kind = staged && session.StageFirst && std::ranges::equal(recordable, std::span{session.Recorded->Actions}.subspan(*session.StageFirst), [](const auto &a, const auto &previous) { return Kind(a) == Kind(previous); });
    const bool single = actions.size() == 1;
    if (same_kind && (!single || action::IsPreview(actions.front()))) {
        if (!single || !action::PatchesPreview(actions.front())) (void)session.History.Restore(*session.StageBase);
    } else if (staged && !recordable.empty()) {
        if (session.StageBase) session.History.Release(*session.StageBase);
        session.StageBase = session.History.Pin();
    }
    auto before = single ? std::optional<store::Snapshot>{} : session.History.Pin();
    const auto first = same_kind ? *session.StageFirst : session.Recorded->Actions.size();
    const auto errors = session.R.Resources.get<action::Errors>().Messages.size();
    bool recorded = false;
    for (auto &a : actions) {
        if (auto *path = SourcePath(a); path && !path->native().starts_with("asset:/")) {
            std::error_code ec;
            const auto absolute = std::filesystem::absolute(*path, ec);
            if (ec) action::Fail(session.R, "Cannot resolve '" + path->string() + "': " + ec.message());
            else *path = session.R.Resources.get<Assets>().Reference(absolute);
        }
        const bool keep = action::IsRecordable(a);
        auto &frame = session.R.Resources.get<FrameState>();
        frame.Scene = ActingScene(session.R);
        const ActionInputs inputs{frame.Scene, frame.DeltaTime, session.R.get<const PlaybackFrame>(state::Document).Value, session.R.get<const TimelinePlayback>(state::Document).CurrentFrame, frame.FixedFrameStep, pass, frame.DisplayFramebufferScale, staged};
        if (session.R.all_of<ActionInputs>(state::Document)) session.R.replace_if_changed(state::Document, inputs);
        else session.R.emplace<ActionInputs>(state::Document, inputs);
        const auto workspace = keep ? std::optional{session.History.CaptureWorkspace()} : std::nullopt;
        frame.GestureOpen |= staged;
        if (session.R.Resources.get<action::Errors>().Messages.size() == errors) Tick(session, a, pass);
        pass = EventPass::Settle;
        if (session.R.Resources.get<action::Errors>().Messages.size() != errors) {
            if (workspace) session.History.ReleaseWorkspace(*workspace);
            if (staged) session.CancelGesture();
            else if (before) {
                (void)session.History.Restore(*before);
                session.ResizeRecorded(first);
            }
            recorded = false;
            break;
        }
        if (!keep) continue;
        if (!recorded) {
            session.ResizeRecorded(first);
            if (staged) session.StageFirst = first;
        }
        session.Recorded->Actions.push_back(std::move(a));
        session.RecordedWorkspaces.push_back(*workspace);
        ++session.Revision;
        recorded = true;
    }
    if (before) session.History.Release(*before);
    return recorded;
}
bool Record(Project &session, action::Action a, EventPass pass, bool staged = false) { return Record(session, std::span{&a, 1}, pass, staged); }
// Applies the actions and commits them as one history step, after finishing any open gesture.
// `keys` also keys the animated properties the step changed, as a user's commit does.
// Returns the committed node, or the present node when nothing was recorded or a selection edit stays open.
std::optional<uint32_t> Step(Project &session, std::span<action::Action> actions, EventPass pass, std::string label, bool keys) {
    auto &r = session.R;
    if (r.all_of<procedure::SelectionEditing>(state::Document) && std::ranges::any_of(actions, [](const auto &a) { return !ContinuesSelection(a); })) session.FinishGesture(EventPass::Settle);
    session.FinishGesture(EventPass::Settle, true);
    if (label.empty()) label = Label(actions.back());
    const bool recorded = Record(session, actions, pass, Is<action::nodes::EditSelection>(actions.front()));
    if (!recorded || r.all_of<procedure::SelectionEditing>(state::Document)) return session.History.Present;
    if (keys) session.RecordKeys();
    const auto committed = session.Commit(std::move(label));
    session.ReleaseGesture();
    return committed;
}
// Applies the posted steps and returns whether there were any.
bool ApplyPosted(Project &session, EventPass pass) {
    auto steps = std::exchange(Steps, {});
    for (auto &a : steps) {
        Step(session, std::span{&a, 1}, pass, {}, false);
        pass = EventPass::Settle;
    }
    return !steps.empty();
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
            return [&]<typename C, typename F>(const action::PatchFields<C, F> &patch) {
                return "Patch " + std::string{state::LeafName<C>()} + ":" + action::FieldAt<C>(patch.Offset).Path;
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
    const bool editable = Editable(Recorded->Actions);
    const bool begun = History.Begin(dir, Encode(Recorded->Actions), std::exchange(RecordedWorkspaces, {}));
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
    if (empty) Record(*this, action::MakeAction(action::io::LoadEmptyScene{}), EventPass::Settle);
    else Record(*this, action::MakeAction(action::io::LoadDefaultScene{}), EventPass::Settle);
    const bool begun = Begin(dir);
    if (!begun) ReturnToLeft(left);
    History.Release(left.Pin);
    Entities.Workspace.Trie.Release(left.Workspace);
    return begun;
}
bool Project::Save() {
    if (!History.Present) return true;
    WaitForRender(R);
    FinishGesture(EventPass::Settle);
    Record(*this, action::MakeAction(action::timeline::SetFrame{R.get<const TimelinePlayback>(state::Document).CurrentFrame}), EventPass::Settle);
    Commit("Playback position");
    return History.Save();
}
std::unique_ptr<Recording, void (*)(Recording *)> NewRecording() {
    return {new Recording, [](Recording *recording) { delete recording; }};
}
std::expected<void, std::string> RunEncoded(Project &session, store::History &source, std::span<const std::byte> actions, std::span<const uint32_t> workspaces) {
    return RunRecorded(session, source, Decode(actions), workspaces);
}
bool EditableEncoded(std::span<const std::byte> actions) { return Editable(Decode(actions)); }
void DiscardPostedSteps() { Steps.clear(); }

std::optional<uint32_t> Do(Project &session, action::Action a, std::string label) {
    WaitForRender(session.R);
    return Step(session, std::span{&a, 1}, EventPass::Frame, std::move(label), true);
}
EditDraft &DraftOf(Project &session, uint32_t node) {
    if (!session.Recorded->Draft || session.Recorded->Draft->Node != node || session.Recorded->Draft->Revision != session.History.Revision) session.Recorded->Draft = EditDraft{node, session.History.Revision, Decode(session.History.ActionsOf(session.History.Nodes[node]))};
    return *session.Recorded->Draft;
}
void Project::ApplySteps() {
    WaitForRender(R);
    ApplyPosted(*this, EventPass::Frame);
}
void Project::ResizeRecorded(size_t count) {
    while (RecordedWorkspaces.size() > count) {
        History.ReleaseWorkspace(RecordedWorkspaces.back());
        RecordedWorkspaces.pop_back();
    }
    Recorded->Actions.resize(count);
}
void Project::RecordKeys() {
    if (!R.get<const Animations>(state::Document).Record) return;
    const auto seconds = animation::FrameSeconds(R, R.get<const TimelinePlayback>(state::Document).CurrentFrame);
    if (animation::AnyChanged(R, seconds)) Record(*this, action::MakeAction(action::animation::RecordChanged{}), EventPass::Settle);
}
uint32_t Project::Commit(std::string label, std::optional<uint32_t> replace) {
    if (!Recorded->Actions.empty() && Is<action::nodes::EditSelection>(Recorded->Actions.front())) label = "Edit selection";
    // The step's new Source nodes take the state it ends with, which the node it commits records.
    FinishSourceStates(R);
    EndCookStep(R);
    R.Resources.get<MeshStore>().IndexHistory();
    const bool editable = Editable(Recorded->Actions);
    auto bytes = Encode(Recorded->Actions);
    const bool in_place = replace && History.Nodes[*replace].Children.empty();
    const auto created = uint32_t(History.Nodes.size());
    const auto node = in_place ? History.Replace(*replace, std::move(bytes), std::exchange(RecordedWorkspaces, {})) : History.Commit(std::move(label), std::move(bytes), std::exchange(RecordedWorkspaces, {}));
    EndMeshStates(R);
    if (in_place || node == created) History.Nodes[node].Editable = editable;
    Editing.reset();
    ResizeRecorded(0);
    History.Evict(MemoryCap);
    return node;
}
void Project::FinishGesture(EventPass pass, bool keep_selection) {
    if (!HasStaged()) return;
    if (keep_selection && R.all_of<procedure::SelectionEditing>(state::Document)) {
        EndGesture(pass);
        if (StageBase) History.Release(*StageBase);
        StageBase.reset();
        StageFirst.reset();
        return;
    }
    if (R.all_of<procedure::SelectionEditing>(state::Document)) Record(*this, action::MakeAction(action::nodes::FinishSelection{}), pass);
    if (Editing) {
        R.Resources.get<FrameState>().GestureOpen = false;
        Settle(pass);
        Commit(History.Nodes[*Editing].Label, Editing);
    } else {
        std::vector<std::string> names;
        for (size_t i = GestureStart; i < Recorded->Actions.size(); ++i) {
            if (auto name = Label(Recorded->Actions[i]); !std::ranges::contains(names, name)) names.push_back(std::move(name));
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
    R.Resources.get<FrameState>().GestureOpen = false;
    Settle(pass);
}
void Project::RestageDraft() {
    if (!Recorded->Draft) return;
    if (Editing != Recorded->Draft->Node && !EditNode(Recorded->Draft->Node)) return;
    if (GestureBase) {
        if (!History.Restore(*GestureBase)) return;
    } else {
        GestureBase = History.Pin();
        GestureStart = Recorded->Actions.size();
    }
    ResizeRecorded(GestureStart);
    // The run takes a copy so the draft keeps its values for the next change.
    auto recorded_actions = Decode(Encode(Recorded->Draft->RecordedActions));
    const auto workspaces = History.WorkspacesOf(History.Nodes[Recorded->Draft->Node]);
    if (!RunRecorded(*this, History, recorded_actions, workspaces)) {
        CancelGesture();
        return;
    }
    for (auto &recorded_action : recorded_actions) Recorded->Actions.push_back(std::move(recorded_action));
    for (auto id : workspaces) {
        History.RetainWorkspace(id);
        RecordedWorkspaces.push_back(id);
    }
    ++Revision;
}
void Project::ReleaseGesture() {
    R.Resources.get<FrameState>().GestureOpen = false;
    EndCookStep(R);
    if (GestureBase) History.Release(*GestureBase);
    if (StageBase) History.Release(*StageBase);
    GestureBase.reset();
    StageBase.reset();
    StageFirst.reset();
}
void Project::CancelGesture() {
    if (!GestureBase) return;
    ResizeRecorded(0);
    ClearInteraction();
    (void)History.Restore(*GestureBase);
    ReleaseGesture();
    if (const auto node = std::exchange(Editing, std::nullopt)) History.Navigate(*node);
    Recorded->Draft.reset();
    ++Revision;
}
void Project::Frame() {
    auto emitted = std::exchange(Emitted, {});
    const bool commit_requested = std::exchange(CommitRequested, false), cancel_requested = std::exchange(CancelRequested, false);
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
    if (cancel_requested && HasStaged()) {
        CancelGesture();
        Settle();
        pass = EventPass::Settle;
    }
    if (emitted && !emitted->first.empty()) {
        auto &[actions, phase] = *emitted;
        if (phase == action::Phase::Cancel) {
            std::optional<bool> duplicate;
            for (const auto &recorded_action : Recorded->Actions) {
                if (Is<action::object::Duplicate>(recorded_action)) duplicate = false;
                if (Is<action::object::DuplicateLinked>(recorded_action)) duplicate = true;
            }
            CancelGesture();
            if (duplicate) {
                Record(*this, *duplicate ? action::MakeAction(action::object::DuplicateLinked{}) : action::MakeAction(action::object::Duplicate{}), EventPass::Settle, true);
            }
            for (const auto &a : actions) {
                Tick(*this, a, pass);
                pass = EventPass::Settle;
            }
        } else if (phase == action::Phase::Record) {
            Step(*this, actions, pass, {}, true);
        } else {
            if (R.all_of<procedure::SelectionEditing>(state::Document) && std::ranges::any_of(actions, [](const auto &a) { return !ContinuesSelection(a); })) {
                FinishGesture(EventPass::Settle);
            }
            Record(*this, actions, pass, true);
        }
        pass = EventPass::Settle;
    }
    if (commit_requested && HasStaged()) {
        FinishGesture(pass, true);
        pass = EventPass::Settle;
    }
    if (!HasStaged() && ApplyPosted(*this, pass)) pass = EventPass::Settle;
    if (pass == EventPass::Frame) Settle();
}
bool Project::EditNode(uint32_t node) {
    const auto parent = History.Nodes[node].Parent;
    if (!parent) return false;
    Navigate(*parent);
    if (History.Present != parent) return false;
    Editing = node;
    return true;
}
} // namespace project
