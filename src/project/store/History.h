#pragma once

#include "project/store/LiveTrie.h"
#include "project/store/Log.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Branching history over registered tries with opaque action bytes.
namespace store {
// Versions use track registration order.
struct Snapshot {
    std::vector<Version> Versions;
};

enum class RecordKind : uint8_t {
    Root, // Replace history with the current baseline and optional saved baseline.
    Action,
    Navigate, // Restore Parent when reopening.
};

struct HistoryNode {
    int Parent{-1};
    std::vector<int> Children;
    std::vector<std::byte> Action;
    std::string Label;
    int Depth{};
    bool ReplayBaseline{true};
    std::optional<Snapshot> Hot;
    // Stamps and Roots use track registration order and remain available after eviction.
    std::vector<Stamp> Stamps;
    std::vector<Hash128> Roots;
    uint64_t LastVisited{}; // VisitCounter value at the last navigation to this node.
};

struct HistoryPosition {
    int Node{-1}; // Preferred node when it still matches the stored state.
    std::vector<Stamp> Stamps;
    std::vector<Hash128> Roots;
};

struct HistoryStats {
    uint64_t OwnedBytes{}, Nodes{}, AliasedNodes{};
    size_t HotNodes{}, ColdNodes{};
    uint64_t SharedNodeBytes{}, HashBytes{}, ManifestBytes{}, PendingBytes{}, PeakPendingBytes{};
    // Vector capacities and estimated hash-map allocations, excluding live data and allocator overhead.
    uint64_t MetadataBytes{};
    // Includes the shared node pool once; concurrent histories share this contribution.
    uint64_t RetainedBytes() const { return OwnedBytes + SharedNodeBytes + HashBytes + ManifestBytes + MetadataBytes + PendingBytes; }
};

struct History {
    struct Hooks {
        // Complete all writes caused by replay before returning.
        std::function<void(const std::vector<std::byte> &action)> Replay{};
        // Run before restoring any track.
        std::function<void()> BeforeRestore{};
        // Run after every track is restored and before enabling LiveTrie::Write.
        std::function<void()> AfterTracks{};
        // Reconcile Derived state after AfterTracks with LiveTrie::Write enabled.
        std::function<void()> AfterRestore{};
    };

    History() = default;
    ~History() { Close(); }
    void Track(LiveTrie &, std::string name, int phase);
    // Increment when component or action encoding changes, including same-size layout changes.
    uint32_t SchemaRevision{};

    // Create a history with live state as node 0 in a directory other than the currently open project.
    // Return false on failure and preserve the current tree.
    bool Begin(const std::filesystem::path &dir);
    // Restore position, or the latest working position when omitted, after validating the format and target.
    // Return false on failure and preserve live state and the current tree.
    bool Open(const std::filesystem::path &dir, const HistoryPosition *position = nullptr);
    int FindPosition(const HistoryPosition &) const;
    bool Close();
    // Continue writing the current records at their copied or moved directory.
    bool Relocate(const std::filesystem::path &dir);

    // Enqueue live state and return its node ID, reusing a matching present, child, or parent node.
    int Commit(std::string label, std::vector<std::byte> action);
    // Call after CPU and GPU writes complete.
    void SettleHashes();
    // Restore node using its snapshot or stored data and update Present.
    void Navigate(int node);
    bool CanUndo() const { return Present > 0; }
    bool CanRedo() const { return Present >= 0 && !Nodes[Present].Children.empty(); }
    void Undo() {
        if (CanUndo()) Navigate(Nodes[Present].Parent);
    }
    // Redo follows the most recently created child.
    void Redo() {
        if (CanRedo()) Navigate(Nodes[Present].Children.back());
    }
    // Discard uncommitted edits and restore the present node.
    void Revert();
    // Pin live state independently of the tree.
    Snapshot Pin();
    void Restore(const Snapshot &);
    void Release(Snapshot &);

    // Record the present node, flush streams to the OS cache, and return whether all writes and flushes succeeded.
    bool Save();
    // Retain live state and the optional saved state as replay baselines after flushing one replacement record.
    // Preserve content records for deduplication.
    // The saved state must reference this history's content records.
    bool Clear(const HistoryPosition *saved = nullptr);

    // Eviction preserves snapshots for the present and this many ancestors, through its replay baseline.
    static constexpr int UndoWindow = 16;
    // Flush pending writes and evict snapshots until OwnedBytes meets the cap or only protected nodes remain.
    void Evict(uint64_t owned_bytes_cap);
    // Include queued records in the content-log size.
    uint64_t LogBytes() const { return LeafLog.Size + NodeLog.Size; }

    // Serialize a node with a cached snapshot for byte comparisons.
    std::vector<std::byte> Materialize(int node) const;
    std::vector<std::byte> MaterializeLive();
    // Replay from the baseline, returning an error or differing track name, or empty on success.
    // Restore original live state on failure.
    std::string Replay(int node);
    // Replay one action from its parent, returning an error or differing track name, or empty on success.
    // Restore the present node before returning.
    std::string ValidateReplay(int node);

    HistoryStats Stats() const;
    bool Check(std::string &why) const;
    // Update hashes and verify trie structure and hashes against live data.
    bool Audit(std::string &why);
    // Return and clear IntegrityError, or return the writer error when IntegrityError is empty.
    std::string TakeIntegrityError();

    struct Tracked {
        LiveTrie *Trie;
        std::string Name;
        // Tries restore in ascending phase, stable within a phase.
        int Phase;
    };
    struct Extent {
        uint64_t Offset{};
        uint32_t Size{};
    };
    struct ContentLog {
        const char *Name;
        size_t File;
        std::unordered_map<Hash128, Extent, Hash128Hasher> Idx;
        uint64_t Size{}; // Includes queued records.
    };

    std::vector<Tracked> Tracks;
    std::vector<size_t> Order; // Track indices in restore order.
    std::vector<HistoryNode> Nodes;
    int Present{-1};
    uint64_t VisitCounter{}, PeakPendingBytes{};
    uint64_t Revision{}; // changes when nodes are added, replaced, or removed
    Hooks Callbacks;
    std::filesystem::path Dir;
    WriteBehind Log;
    ContentLog LeafLog{"leaves.log", 0, {}, 0}, NodeLog{"nodes.log", 1, {}, 0};
    std::vector<std::vector<std::byte>> Pending{3}; // One append buffer per log.
    std::string IntegrityError;
};
} // namespace store
