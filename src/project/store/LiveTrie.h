#pragma once

#include "project/store/Blob.h"
#include "project/store/Manifest.h"

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Versioned slots over bytes and hashes the owner supplies.
// Capture a slot before mutating it and rehash after CPU and GPU writes complete.
// Pinned versions share trie nodes and copies of changed values.
namespace store {

enum class NodeKind : uint8_t {
    Interior,
    Aliased, // equal to live over the node's whole range, any level
    Owned, // A copied leaf value.
    Absent,
};
struct Node {
    uint32_t Refs;
    NodeKind Kind;
    Blob Value; // Owned only
    Node **Children; // Interior only
    Hash128 Hash; // Owned only: the content hash of Value
};

// Release each pinned version through its LiveTrie.
struct Version {
    Node *Root{};
    Stamp S{};
};

// One slot the owner applies while restoring or loading a version.
struct SlotChange {
    uint64_t Slot{};
    // The version's value, moved out of the trie. The owner takes it when applying and leaves it when Unchanged.
    Blob Incoming{};
    bool Erase{}; // The version holds no value at Slot.
    bool MaybeEqual{}; // The settled hash matches Incoming, so the owner compares bytes before replacing.
    bool Default{}; // Incoming hashes as a default page.
    // The owner fills these while applying.
    Blob Old{}; // The displaced live value when WasPresent and not Unchanged.
    bool WasPresent{};
    bool Unchanged{}; // Live already held Incoming, or nothing was present to erase.
};
struct RestorePlan {
    uint64_t Length{};
    std::vector<SlotChange> Changes;
    Node *Target{}; // The version root, or null for a load.
    Hash128 Hash{}; // The version's recorded state hash.
};

// A run of slots in a version: one owned value, or Count slots that alias live data.
struct MaterializedRun {
    uint64_t Slot{}, Count{};
    const Blob *Owned{};
};

// Whether live bytes already equal the incoming encoded value.
inline bool Unchanged(std::span<const std::byte> current, std::span<const std::byte> incoming) {
    return current.size() == incoming.size() && (current.empty() || std::memcmp(current.data(), incoming.data(), current.size()) == 0);
}

// Process-wide backing allocation, including cached node/child-array blocks.
uint64_t SharedNodePoolBytes();

struct TrieStats {
    uint64_t Nodes{}, AliasedNodes{}, OwnedSlots{}, OwnedBytes{};
};

struct LiveTrie {
    // Capacity is Fanout^levels slots.
    // page_bytes is the size of zero-default byte pages, or zero for records.
    LiveTrie(uint32_t levels, uint32_t page_bytes = 0);
    ~LiveTrie();
    LiveTrie(const LiveTrie &) = delete;
    LiveTrie &operator=(const LiveTrie &) = delete;

    // Capture pages of [first, first + count) that pinned versions still alias and mark them dirty.
    // contents spans whole pages of live storage.
    void Write(uint64_t first, uint64_t count, std::span<const std::byte> contents);
    // Whether pinned versions alias slot, so its value must be captured before mutation.
    bool Uncaptured(uint64_t slot) const;
    // Record slot's live value for pinned versions. Absent slots pass nullopt.
    void Capture(uint64_t slot, std::optional<Blob>);
    // Schedule [first, first + count) for rehashing.
    void MarkDirty(uint64_t first, uint64_t count);
    bool IsDirty(uint64_t slot) const { return slot < SlotHashes.size() && SlotHashes[slot].Dirty; }

    // Slots awaiting Rehash. Rehash each, then ClearDirty.
    std::span<const uint64_t> Dirty() const { return DirtySlots; }
    // Hash slot's live bytes. Absent slots pass nullopt.
    void Rehash(uint64_t slot, std::optional<std::span<const std::byte>> bytes);
    void ClearDirty() { DirtySlots.clear(); }
    // Rehash dirty pages from live storage.
    void SettleFlat(std::span<const std::byte> contents);

    // Require settled hashes.
    Stamp CurrentStamp(uint64_t length) const;
    Version Pin(uint64_t length);
    void Release(Version &);

    // Plan the slot changes that restore v. Apply them, then CommitRestore.
    // Requires settled hashes. No other trie operation may run between the two calls.
    RestorePlan PlanRestore(const Version &);
    // Swap displaced values into the version's nodes, rehash changed slots, and return whether the hash matches the recorded stamp.
    bool CommitRestore(RestorePlan &&);
    // Plan validated changes, erasing slots with zero hashes. Apply them, then CommitLoad.
    RestorePlan PlanLoad(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves);
    // Capture displaced values for pinned versions and mark loaded slots dirty.
    void CommitLoad(RestorePlan &&);

    // Present runs of a version in slot order, bounded by its length.
    std::vector<MaterializedRun> Materialize(const Version &) const;
    // Requires settled hashes. Return the manifest root for live data.
    Hash128 ManifestRoot(uint64_t length);
    // Requires current slot hashes and child-level hashes.
    ManifestChildren ChildrenAt(uint32_t level, uint64_t index, uint64_t length) const;
    uint64_t SlotsFor(uint64_t length) const { return PageBytes ? (length + PageBytes - 1) / PageBytes : length; }

    // Restore and load append slot indices when CollectChanged is set.
    // The consumer clears ChangedSlots after use.
    bool CollectChanged{};
    std::vector<uint64_t> ChangedSlots;
    std::vector<uint64_t> TakeChanged() { return std::exchange(ChangedSlots, {}); }

    const TrieStats &Stats() const { return S; }
    // Checks trie structure and that all pinned Aliased nodes are reachable from the present.
    bool Check(std::string &why, const std::vector<Version> &all_versions) const;
    // Compare settled hashes with the hashes of live non-default slots, given in slot order.
    bool CheckHashes(std::string &why, std::span<const std::pair<uint64_t, Hash128>> live) const;

    enum class SlotState : uint8_t { Unhashed,
                                     Value,
                                     Default };
    struct SlotHash {
        Hash128 H{};
        SlotState State{};
        bool Dirty{};
    };

    const uint32_t Levels, PageBytes;

    TrieStats S;
    Node *Root;

    std::vector<SlotHash> SlotHashes; // grown to the highest touched slot
    std::vector<uint64_t> DirtySlots;
    struct ManifestNode {
        Hash128 Hash{};
        bool Dirty{};
    };
    struct ManifestLevel {
        std::vector<ManifestNode> Nodes;
        std::vector<uint64_t> Dirty;
    };
    std::vector<ManifestLevel> Manifest;
    uint64_t ManifestSlots{};
    uint64_t Lane0{}, Lane1{}; // Wrapping sums of Term::L0 and Term::L1 over non-default slots.

    bool Busy{}; // Set between a plan and its commit.
    bool ExternalWritesForbidden{}; // Owners assert against writes during track restoration.
};
} // namespace store
