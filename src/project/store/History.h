#pragma once

#include "project/store/LiveTrie.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace project {
struct ComponentPool;
}

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
    uint64_t OwnedBytes{}, SharedNodeBytes{};
    size_t HotNodes{}, ColdNodes{};
};

struct History {
    struct Hooks {
        // Complete all writes caused by replay before returning.
        std::function<void(const std::vector<std::byte> &action)> Replay{};
        // Run before restoring any track.
        std::function<void()> BeforeRestore{};
        // Run after every track is restored and before enabling track writes.
        std::function<void()> AfterTracks{};
        // Reconcile Derived state after AfterTracks with track writes enabled.
        std::function<void()> AfterRestore{};
    };

    ~History() { Close(); }
    // Tracks restore in ascending phase, stable within a phase.
    void Track(Pages &, std::string name, int phase);
    void Track(Records &, std::string name, int phase);
    void Track(project::ComponentPool &, std::string name, int phase);
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

    // Record live state and return its node ID, reusing a matching present, child, or parent node.
    int Commit(std::string label, std::vector<std::byte> action);
    // Record live state as `node`'s new content under its label, drop the node's descendants, and make it present.
    int Replace(int node, std::vector<std::byte> action);
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
    // Flush writes and evict snapshots until OwnedBytes meets the cap or only protected nodes remain.
    void Evict(uint64_t owned_bytes_cap);
    uint64_t LogBytes() const { return LeafLog.Size + NodeLog.Size; }

    // Serialize live state for byte comparisons.
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
    // Return and clear IntegrityError, or return the stream error when IntegrityError is empty.
    std::string TakeIntegrityError();

    enum class Kind : uint8_t { Pages,
                                Records,
                                Pool };
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

    std::vector<Tracked> Tracks;
    std::vector<Pages *> PageTracks;
    std::vector<Records *> RecordTracks;
    std::vector<project::ComponentPool *> PoolTracks;
    std::vector<size_t> Order; // Track indices in restore order.
    std::vector<HistoryNode> Nodes;
    int Present{-1};
    uint64_t VisitCounter{};
    uint64_t Revision{}; // changes when nodes are added, replaced, or removed
    Hooks Callbacks;
    std::filesystem::path Dir;
    ContentLog LeafLog{"leaves.log", 0, {}, 0}, NodeLog{"nodes.log", 1, {}, 0};
    std::array<std::vector<char>, 3> StreamBuffers; // Appends reach the OS on flush or when a buffer fills.
    std::array<std::ofstream, 3> Streams; // Leaf, node, and tree logs, buffered by StreamBuffers.
    std::string IntegrityError;
};
} // namespace store
