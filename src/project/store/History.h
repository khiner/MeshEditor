#pragma once

#include "project/store/LiveTrie.h"

#include <expected>
#include <filesystem>
#include <fstream>
#include <functional>

namespace project {
struct ComponentPool;
struct WorkspacePool;
} // namespace project

// Branching history over registered tracks with opaque action bytes.
namespace store {
struct Pages;
struct Records;

// Versions use track registration order.
struct Snapshot {
    std::vector<Version> Versions;
};

enum class RecordKind : uint8_t {
    Root, // Replace history with the current baseline and optional saved baseline.
    Action,
    Navigate, // Restore Parent when reopening.
    Replace, // Give the node named by Parent new content and drop its descendants.
    State, // Give the node named by Parent the content its replay produced.
    Workspace, // A workspace version referenced by recorded actions.
    Kept, // Record the manifest roots of some tracks in the state with a given hash.
};

struct WorkspaceVersion {
    Stamp S{};
    Hash128 Root{};
    Version Hot{};
    uint32_t Refs{};
};
struct HistoryRange {
    uint64_t First{}, Count{};
    bool empty() const { return Count == 0; }
};
struct HistoryNode {
    std::optional<uint32_t> Parent;
    std::vector<uint32_t> Children;
    std::optional<uint32_t> RedoChild; // The child on the most recently active lineage.
    HistoryRange Actions; // Into the flat encoded action bytes.
    HistoryRange Workspaces; // Into the parallel workspace references.
    std::string Label;
    uint32_t Depth{};
    bool ReplayBaseline{true}; // Replay starts from this node's stored state, and a baseline without state runs its actions from the empty state.
    bool Editable{false}; // Whether any of the node's actions has parameters to edit, set by the project that records them.
    std::optional<Snapshot> Hot;
    // Stamps and Roots use track registration order and remain available after eviction.
    // Both are empty until the node's first visit replays its actions from its nearest ancestor with state.
    std::vector<Stamp> Stamps;
    std::vector<Hash128> Roots;
    uint64_t LastVisited{}; // VisitCounter value at the last navigation to this node.
};

struct HistoryPosition {
    std::optional<uint32_t> Node; // Preferred node when it still matches the stored state.
    std::vector<Stamp> Stamps;
    std::vector<Hash128> Roots;
};

// The hash that identifies the content of tracks with these stamps.
Hash128 StateHash(std::span<const Stamp>);

// Manifest roots of some tracks in the state with a given hash, which Clear retains.
struct KeptState {
    std::vector<uint32_t> Tracks;
    std::vector<Hash128> Roots;
};

struct HistoryStats {
    uint64_t OwnedBytes{}, SharedNodeBytes{};
    size_t HotNodes{}, ColdNodes{};
};

struct History {
    struct Hooks {
        // Complete all writes caused by replay before returning.
        std::function<std::expected<void, std::string>(History &, std::span<const std::byte>, std::span<const uint32_t>)> Replay{};
        // Replace live state with the empty state that baseline actions run from, resolving project files in dir.
        std::function<void(const std::filesystem::path &dir)> Reset{};
        // Run before Save flushes the logs, so the records a reopened history reads are written.
        std::function<void()> BeforeSave{};
        // Run before restoring any track.
        std::function<void()> BeforeRestore{};
        // Run after every track is restored and before enabling track writes.
        std::function<void()> AfterTracks{};
        // Reconcile Derived state after AfterTracks with track writes enabled.
        std::function<std::expected<void, std::string>()> AfterRestore{};
    };

    ~History() { Close(); }
    // Tracks restore in ascending phase, stable within a phase.
    void Track(Pages &, std::string name, int phase);
    void Track(Records &, std::string name, int phase);
    void Track(project::ComponentPool &, std::string name, int phase);
    // Create a history with live state as node 0, reached by `actions` from the empty state, in a directory other than the currently open project.
    // Return false on failure and preserve the current tree.
    bool Begin(const std::filesystem::path &dir, std::vector<std::byte> actions = {}, std::vector<uint32_t> workspaces = {});
    // Restore position, or the latest working position when omitted, after validating the format and target.
    // Build the history tree from the actions log when content logs are absent.
    // Replay the present node from empty state.
    // Return false on failure and preserve live state and the current tree.
    bool Open(const std::filesystem::path &dir, const HistoryPosition *position = nullptr);
    std::optional<uint32_t> FindPosition(const HistoryPosition &) const;
    bool Close();
    // Continue writing the current records at their copied or moved directory.
    bool Relocate(const std::filesystem::path &dir);

    // Record live state and return its node ID, reusing a matching present, child, or parent node.
    uint32_t Commit(std::string label, std::vector<std::byte> actions, std::vector<uint32_t> workspaces = {});
    // Record live state as `node`'s new content under its label, drop the node's descendants, and make it present.
    uint32_t Replace(uint32_t node, std::vector<std::byte> actions, std::vector<uint32_t> workspaces = {});
    // Call after CPU and GPU writes complete.
    void SettleHashes();
    // Restore node from its snapshot, its stored data, or a replay from its nearest ancestor with state, and update Present.
    void Navigate(uint32_t node);
    std::optional<uint32_t> UndoTarget() const { return Present ? Nodes[*Present].Parent : std::nullopt; }
    bool CanUndo() const { return UndoTarget().has_value(); }
    std::optional<uint32_t> RedoTarget() const { return Present ? Nodes[*Present].RedoChild : std::nullopt; }
    bool CanRedo() const { return RedoTarget().has_value(); }
    void Undo() {
        if (const auto node = UndoTarget()) Navigate(*node);
    }
    // Redo follows the most recently active lineage.
    void Redo() {
        if (const auto node = RedoTarget()) Navigate(*node);
    }
    // Discard uncommitted edits and restore the present node.
    void Revert();
    std::span<const std::byte> ActionsOf(const HistoryNode &n) const { return std::span{ActionBytes}.subspan(n.Actions.First, n.Actions.Count); }
    std::span<const uint32_t> WorkspacesOf(const HistoryNode &n) const { return std::span{WorkspaceRefs}.subspan(n.Workspaces.First, n.Workspaces.Count); }
    void StoreActions(HistoryNode &, std::span<const std::byte>, std::span<const uint32_t>);
    void CompactActions();
    uint32_t CaptureWorkspace();
    void RetainWorkspace(uint32_t);
    void ReleaseWorkspace(uint32_t);
    std::expected<void, std::string> RestoreWorkspace(uint32_t);
    void PersistWorkspaces();
    // Pin live state independently of the tree.
    Snapshot Pin();
    std::expected<void, std::string> Restore(const Snapshot &);
    void Release(Snapshot &);
    // The listed tracks' manifest roots in the state whose StateHash over their stamps is `state`, from a node recording it or a kept state, or empty when none holds it.
    std::vector<Hash128> RootsOf(Hash128 state, std::span<const size_t> tracks) const;
    // Record the roots of the state with hash `state` that a node holds, so it resolves after Clear drops that node.
    void Keep(Hash128 state, std::span<const size_t> tracks);
    // Record the roots of the state with hash `state` from the listed tracks' pinned versions of it, persisting the content no recorded state holds.
    void Keep(Hash128 state, std::span<const size_t> tracks, std::span<const Version> versions);

    // Record the present node, flush streams to the OS cache, and return whether all writes and flushes succeeded.
    bool Save();
    // Retain live state and the optional saved state as replay baselines after flushing one replacement record.
    // Preserve content records for deduplication.
    // The saved state must reference this history's content records.
    bool Clear(const HistoryPosition *saved = nullptr);
    // Write the present node and each reachable node's actions, label and parent, in index order, as an actions log, and return those nodes.
    // Return why the tree cannot replay when its root contains state without actions.
    std::expected<std::vector<uint32_t>, std::string> WriteActions(const std::filesystem::path &) const;

    // Eviction preserves snapshots for the present and this many ancestors, through its replay baseline.
    static constexpr uint32_t UndoWindow = 16;
    // Flush writes and evict snapshots until OwnedBytes meets the cap or only protected nodes remain.
    void Evict(uint64_t owned_bytes_cap);
    uint64_t LogBytes() const { return LeafLog.Size + NodeLog.Size; }

    // Serialize live state for byte comparisons.
    std::vector<std::byte> MaterializeLive();
    // Validate replay against retained states for the present node and its ancestors.
    // Replay the present node from its replay baseline, returning an error or differing track name, or empty on success.
    // Restore original live state on failure.
    std::string Replay();
    // Replay the present node's actions from its parent, returning an error or differing track name, or empty on success.
    // Restore the present node before returning.
    std::string ValidateReplay();

    HistoryStats Stats() const;
    bool Check(std::string &why) const;
    // Update hashes and verify trie structure and hashes against live data.
    bool Audit(std::string &why);
    // Return and clear IntegrityError, or return the stream error when IntegrityError is empty.
    std::string TakeIntegrityError();

    enum class Kind : uint8_t { Pages,
                                Records,
                                Pool,
                                Workspace };
    struct Tracked {
        std::string Name;
        int Phase;
        Kind K;
        size_t Index; // Into the vector for K.
    };
    struct Extent {
        uint64_t Offset{};
        uint32_t Size{};
    };
    struct ContentLog {
        const char *Name;
        size_t File;
        std::unordered_map<Hash128, Extent, Hash128Hasher> Idx;
        uint64_t Size{};
    };

    project::WorkspacePool *Workspace{};
    std::vector<WorkspaceVersion> Workspaces;
    uint32_t WrittenWorkspaces{};
    std::unordered_map<Hash128, KeptState, Hash128Hasher> Kept;
    std::vector<Tracked> Tracks;
    std::vector<Pages *> PageTracks;
    std::vector<Records *> RecordTracks;
    std::vector<project::ComponentPool *> PoolTracks;
    std::vector<size_t> Order; // Track indices in restore order.
    std::vector<std::byte> ActionBytes;
    std::vector<uint32_t> WorkspaceRefs;
    std::vector<HistoryNode> Nodes;
    std::optional<uint32_t> Present;
    uint64_t VisitCounter{};
    uint64_t Revision{}; // changes when nodes are added, replaced, or removed
    Hooks Callbacks;
    std::filesystem::path Dir;
    ContentLog LeafLog{"leaves.log", 0, {}, 0}, NodeLog{"nodes.log", 1, {}, 0};
    static constexpr const char *TreeLogName{"tree.log"}, *ActionsLogName{"actions.log"};
    std::array<std::vector<char>, 3> StreamBuffers; // Appends reach the OS on flush or when a buffer fills.
    std::array<std::ofstream, 3> Streams; // Leaf, node, and tree logs, buffered by StreamBuffers.
    std::string IntegrityError;
};

// Reads slots of recorded states from the content logs, flushed when the reader opens.
struct StateReader {
    explicit StateReader(History &);
    bool Open() const { return Leaves && Nodes; }
    // The slot's encoded value in the state where the track's manifest root is `root`, empty for an absent or default slot, or none for missing records.
    std::optional<std::vector<std::byte>> Read(size_t track, Hash128 root, uint64_t slot);
    // The ascending slots whose content in that state differs from the track's live content, or none when its records are missing.
    std::optional<std::vector<uint64_t>> DifferingSlots(size_t track, Hash128 root);

    History &Hist;
    std::ifstream Leaves, Nodes;
};
} // namespace store
