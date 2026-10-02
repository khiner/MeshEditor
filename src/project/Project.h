#pragma once

#include "File.h"
#include "ProcessEvents.h"
#include "action/Action.h"
#include "action/ActionDrain.h"
#include "project/EntityStore.h"
#include "project/store/History.h"

#include <memory>
#include <span>

namespace action::mesh {
struct InsetPreviewCache;
}

namespace project {
// An action's name, and for a field write the component and field it names.
std::string Label(const action::Action &);

// A project archive holds the working directory, and an actions archive holds the recorded actions and their assets.
// Both archive forms save one final workspace snapshot.
enum class ArchiveForm : uint8_t { Project,
                                   Actions };
// Write the actions, their assets, and the final workspace, or return why it cannot.
std::expected<void, std::string> WriteActionsArchive(const store::History &, const std::filesystem::path &path, std::span<const std::byte> workspace);

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
    // A project archive also records the saved position and workspace.
    bool SaveArchive(const std::filesystem::path &, ArchiveForm, std::span<const std::byte> workspace);
    // Obtain user confirmation before replacing an existing project directory.
    bool SaveAs(const std::filesystem::path &directory, std::span<const std::byte> workspace);
    bool RevertSaved();
    bool ClearHistory();
    bool Close();

    std::optional<uint32_t> Do(action::Action, std::string label = {});
    void Frame(action::Drained);
    void Enqueue(action::Action a) { Deferred.push_back(std::move(a)); }
    bool HasStaged() const { return GestureBase.has_value(); }
    void CancelGesture();
    void Settle(EventPass = EventPass::Frame);
    void Navigate(uint32_t node);
    void Undo();
    void Redo();
    bool Replay();
    void RequestNavigate(uint32_t node) { Navigation = node; }
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
        bool Staged{};
        bool PreviewSeed{};
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
    // The action being applied is a staged preview, so mesh operators draw their output without replacing the mesh.
    bool Previewing{false};
    std::unique_ptr<action::mesh::InsetPreviewCache> InsetPreview;
    std::optional<uint32_t> Navigation;
    std::optional<uint32_t> Editing; // The node the open gesture replaces on commit

    struct EditDraft {
        uint32_t Node;
        uint64_t Revision; // The history revision the actions were decoded at.
        std::vector<RecordedAction> RecordedActions;
    };
    std::optional<EditDraft> Draft;
    bool RestageRequested{false};
    // The draft of `node`, decoded anew when the node or the history changed.
    EditDraft &DraftOf(uint32_t node);
    // Whether any of the node's actions has parameters to edit.
    bool Editable(uint32_t node) const;
    // Re-runs the draft on its node's parent at frame end.
    // The commit that follows replaces a leaf node and forks a node with children.
    void RequestRestage() { RestageRequested = true; }

    void Tick(const action::Action &, EventPass = EventPass::Frame, bool staged = false);
    // Applies the action and records it with the frame inputs it ran under.
    bool Record(action::Action, EventPass, bool staged = false);
    std::expected<void, std::string> RunRecorded(std::span<const RecordedAction>);
    // Records the actions as a new node, replacing `replace` when it has no children and adding a sibling otherwise.
    uint32_t Commit(std::string label, std::optional<uint32_t> replace = {});
    void FinishGesture(EventPass);
    // Removes the drag-start records a gesture's updates made.
    void EndGesture(EventPass);
    // Moves to `node`'s parent so the open gesture replaces the node on commit.
    bool EditNode(uint32_t node);
    void RestageDraft();
    // Keys changed animated properties before a user commit while recording.
    void RecordKeys();
    void ReleaseGesture();
    void ClearInteraction();
    // Replaces the document with the empty scene.
    void ClearDocument();
    void AfterRestore();
    state::Scene &R;
    state::Entity Viewport{state::Null};
};

Project &Session(state::Scene &);
} // namespace project
