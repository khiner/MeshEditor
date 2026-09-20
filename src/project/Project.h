#pragma once

#include "File.h"
#include "ProcessEvents.h"
#include "action/Action.h"
#include "action/ActionDrain.h"
#include "project/EntityStore.h"
#include "project/store/History.h"

#include <span>

namespace project {
// An action's name, and for a field write the component and field it names.
std::string Label(const action::Action &);

// Construct before engine initialization so all document entities use the versioned allocator.
// Close history before tearing down the scene's stores.
struct Project {
    explicit Project(state::Scene &);
    ~Project();
    void TrackStores(state::Entity viewport);
    bool Begin(const std::filesystem::path &);
    bool New(const std::filesystem::path &, bool empty = true);
    bool Open(const std::filesystem::path &working, const std::filesystem::path &saved = {});
    bool Save();
    bool SaveArchive(const std::filesystem::path &, std::span<const std::byte> workspace = {});
    // Obtain user confirmation before replacing an existing project directory.
    bool SaveAs(const std::filesystem::path &directory, std::span<const std::byte> workspace = {});
    bool RevertSaved();
    bool ClearHistory();
    bool Close();

    int Do(action::Action, std::string label = {});
    void Frame(action::Drained);
    void Enqueue(action::Action a) { Deferred.push_back(std::move(a)); }
    bool HasStaged() const { return GestureBase.has_value(); }
    void CancelGesture();
    void Settle(EventPass = EventPass::Frame);
    void Navigate(int node);
    void Undo();
    void Redo();
    bool Replay();
    void RequestNavigate(int node) { Navigation = node; }
    bool Audit(std::string &why);

    store::History History;
    File::DirectoryLock DirectoryLock;
    std::filesystem::path SavedPath;
    std::vector<std::byte> RestoredWorkspace;
    EntityStore Entities;
    uint64_t MemoryCap{64ull << 20};
    uint64_t Revision{};

    struct ReplayInputs {
        float DeltaTime{}, PlaybackFrame{};
        int CurrentFrame{};
        bool FixedFrameStep{};
        EventPass Pass{EventPass::Frame};
    };
    struct RecordedAction {
        ReplayInputs Inputs;
        action::Action Action;
    };
    std::vector<RecordedAction> RecordedActions;
    std::vector<action::Action> Deferred;
    std::optional<size_t> StageFirst;
    std::optional<store::Snapshot> GestureBase;
    size_t GestureStart{};
    std::optional<int> Navigation;
    std::optional<int> Editing; // The node the open gesture replaces on commit

    struct EditDraft {
        int Node;
        uint64_t Revision; // The history revision the actions were decoded at.
        std::vector<RecordedAction> RecordedActions;
    };
    std::optional<EditDraft> Draft;
    bool RestageRequested{false};
    // The draft of `node`, decoded anew when the node or the history changed.
    EditDraft &DraftOf(int node);
    // Whether any of the node's actions has parameters to edit.
    bool Editable(int node) const;
    // Re-runs the draft on its node's parent at frame end.
    // The commit that follows replaces a leaf node and forks a node with children.
    void RequestRestage() { RestageRequested = true; }

    void Tick(const action::Action &, EventPass = EventPass::Frame);
    // Applies the action and records it with the frame inputs it ran under.
    bool Record(action::Action, EventPass, bool staged = false);
    void RunRecorded(std::span<const RecordedAction>);
    // Records the actions as a new node, replacing `replace` when it has no children and adding a sibling otherwise.
    int Commit(std::string label, std::optional<int> replace = {});
    void FinishGesture(EventPass);
    // Removes the drag-start records a gesture's updates made.
    void EndGesture(EventPass);
    // Moves to `node`'s parent so the open gesture replaces the node on commit.
    void EditNode(int node);
    void RestageDraft();
    // Keys changed animated properties before a user commit while recording.
    void RecordKeys();
    void ReleaseGesture();
    void ClearInteraction();
    void AfterRestore();
    state::Scene &R;
    state::Entity Viewport{state::Null};
};

Project &Session(state::Scene &);
} // namespace project
