#pragma once

#include "File.h"
#include "ProcessEvents.h"
#include "project/EntityStore.h"
#include "project/ProjectSchema.h"
#include "project/store/History.h"

namespace project {
struct Project;
// The recorded actions and the edit draft, which only Recording.cpp can complete because they hold the full action type.
struct Recording;
std::unique_ptr<Recording, void (*)(Recording *)> NewRecording();

// These take actions in the encoded form that history nodes store.
// Applies the encoded actions with their recorded workspaces restored from `source`.
std::expected<void, std::string> RunEncoded(Project &, store::History &source, std::span<const std::byte> actions, std::span<const uint32_t> workspaces);
// Whether any of the encoded actions has a value to edit.
bool EditableEncoded(std::span<const std::byte> actions);
// Empties the queue of posted steps.
void DiscardPostedSteps();

// A project archive contains the working directory; an actions archive contains actions, their workspace pages, and their assets.
// Both archive forms save one final workspace snapshot.
enum class ArchiveForm : uint8_t { Project,
                                   Actions };
// Write the actions, their workspace pages and assets, and the final workspace, or return why it cannot.
std::expected<void, std::string> WriteActionsArchive(const store::History &, const std::filesystem::path &path, std::span<const std::byte> workspace);

// Construct before engine initialization so all document entities use the versioned allocator.
// Close history before tearing down the scene's stores.
struct Project {
    explicit Project(state::Registry &);
    ~Project();
    void TrackStores();
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

    // Applies the frame's emitted gesture and requests, then the posted steps when no gesture is open.
    void Frame();
    // Applies the posted steps now, each as its own history step, after finishing any open gesture.
    void ApplySteps();
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

    std::unique_ptr<Recording, void (*)(Recording *)> Recorded;
    std::vector<uint32_t> RecordedWorkspaces;
    std::optional<size_t> StageFirst;
    std::optional<store::Snapshot> GestureBase;
    std::optional<store::Snapshot> StageBase; // The state before the current stage's first step, which each of its previews reapplies from
    std::optional<store::Snapshot> PlaybackBase; // The state the last recorded action left, while playback cooks graphs that read the frame on it
    bool Returning{}; // A restore returns to the playback base, which leaves the inputs of modal solves, posted steps and interaction in place
    size_t GestureStart{};
    bool Replaced{}; // A step of the open gesture replaced the recorded actions of the step before it
    std::optional<uint32_t> Navigation;
    std::optional<uint32_t> Editing; // The node the open gesture replaces on commit

    bool RestageRequested{false};
    // Re-runs the draft on its node's parent at frame end.
    // The commit that follows replaces a leaf node and forks a node with children.
    void RequestRestage() { RestageRequested = true; }

    void ResizeRecorded(size_t);
    // Records the actions as a new node, replacing `replace` when it has no children and adding a sibling otherwise.
    uint32_t Commit(std::string label, std::optional<uint32_t> replace = {});
    void FinishGesture(EventPass, bool keep_selection = false);
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
    state::Registry &R;

private:
    // The document left for another, pinned, with the workspace it had.
    struct LeftDocument {
        store::Snapshot Pin;
        store::Version Workspace;
    };
    LeftDocument LeaveDocument();
    // Restores the left document after a failed switch.
    void ReturnToLeft(const LeftDocument &);
};

Project &Session(state::Registry &);
} // namespace project
