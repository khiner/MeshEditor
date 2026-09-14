#pragma once

#include "project/store/Blob.h"
#include "project/store/Manifest.h"

#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Versioned slots over externally owned storage.
// Call Write before mutating live data and complete CPU/GPU writes before updating hashes or pinning a version.
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

// Referenced storage must outlive the trie.
struct Live {
    uint32_t PageBytes{}; // Zero for records, or the page size for zero-default byte buffers.
    std::function<uint64_t()> Length{}; // Bytes for buffers, slot count for records.
    // Called after every restore, including when the length is unchanged.
    std::function<void(uint64_t)> SetLength{};
    std::function<bool(uint64_t)> Present{};
    // The returned span remains valid until the next read or mutation.
    std::function<std::span<const std::byte>(uint64_t)> Read{};
    // Take ownership of incoming and return the previous value, or an empty Blob when !was_present.
    std::function<Blob(uint64_t slot, Blob incoming, bool &was_present)> Replace{};
    // Remove a present value and return its owned copy.
    std::function<Blob(uint64_t)> Erase{};
    // Omit to scan slot indices with Present.
    std::function<void(const std::function<void(uint64_t)> &)> ForEachPresent{};
    // Typed CPU values retain their native representation between hot versions.
    std::function<Blob(uint64_t)> Copy{};
    std::function<std::span<const std::byte>(const Blob &)> Encode{};
    // Entity generation changes prevent reuse even when component bytes match.
    std::function<bool(uint64_t)> Reusable{};
    std::function<void(const std::function<void(uint64_t)> &)> ForEachNonReusable{};
};

inline uint64_t SlotsFor(const Live &live, uint64_t length) {
    return live.PageBytes ? (length + live.PageBytes - 1) / live.PageBytes : length;
}
inline Blob Capture(const Live &live, uint64_t slot) { return live.Copy ? live.Copy(slot) : CopyBlob(live.Read(slot)); }
inline std::span<const std::byte> Encoded(const Live &live, const Blob &value) { return value.Destroy && live.Encode ? live.Encode(value) : value.View(); }
inline bool DefaultAt(const Live &live, uint64_t slot) {
    return !live.Present(slot) || (live.PageBytes && IsZero(live.Read(slot)));
}
inline bool DefaultValue(const Live &live, const Blob &value) {
    return live.PageBytes && (value.Size != live.PageBytes || IsZero(value.View()));
}
inline bool Equals(const Live &live, uint64_t slot, const Blob &value) {
    if ((live.Reusable && !live.Reusable(slot)) || !live.Present(slot)) return false;
    const auto bytes = live.Read(slot);
    const auto wanted = Encoded(live, value);
    return bytes.size() == wanted.size() && (bytes.empty() || std::memcmp(bytes.data(), wanted.data(), bytes.size()) == 0);
}
inline void ForEachPresent(const Live &live, const std::function<void(uint64_t)> &fn) {
    if (live.ForEachPresent) live.ForEachPresent(fn);
    else {
        for (uint64_t s = 0, n = SlotsFor(live, live.Length()); s < n; ++s)
            if (live.Present(s)) fn(s);
    }
}

// Process-wide backing allocation, including cached node/child-array blocks.
uint64_t SharedNodePoolBytes();

struct TrieStats {
    uint64_t Nodes{}, AliasedNodes{}, OwnedSlots{}, OwnedBytes{};
};

struct LiveTrie {
    // Capacity is Fanout^levels slots.
    // slot_bytes records the fixed payload size, or zero for variable/empty slots, in the disk format descriptor.
    LiveTrie(Live, uint32_t levels, uint32_t slot_bytes = 0);
    ~LiveTrie();
    LiveTrie(const LiveTrie &) = delete;
    LiveTrie &operator=(const LiveTrie &) = delete;

    // Capture [first, first + count) before mutation and mark its hashes dirty.
    void Write(uint64_t first, uint64_t count);
    // Recompute dirty hashes after CPU and GPU writes complete.
    void SettleHashes();
    // Update dirty hashes and return the live hash and length.
    Stamp CurrentStamp();
    // Update dirty hashes and pin live state.
    Version Pin();
    void Release(Version &);
    // Restore v and return whether the resulting hash matches its recorded hash.
    // On failure, live state contains the restoration result.
    bool Restore(const Version &);
    bool Differs(const Version &) const;
    // Visit present slots in order with byte spans valid for the callback duration.
    void Materialize(const Version &, const std::function<void(uint64_t slot, std::span<const std::byte>)> &) const;
    // Apply validated changes, erasing slots with zero hashes.
    // Apply staged replacements before calling SettleHashes.
    void LoadChanges(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves);
    // Update hashes and return the manifest root for live data.
    Hash128 ManifestRoot();
    // Requires current slot hashes and child-level hashes.
    ManifestChildren ChildrenAt(uint32_t level, uint64_t index) const;
    uint64_t SlotsFor(uint64_t length) const { return store::SlotsFor(L, length); }

    // Restore and LoadChanges append slot indices when CollectChanged is set.
    // The consumer clears ChangedSlots after use.
    bool CollectChanged{};
    std::vector<uint64_t> ChangedSlots;
    std::vector<uint64_t> TakeChanged() { return std::exchange(ChangedSlots, {}); }

    const TrieStats &Stats() const { return S; }
    uint64_t HashStorageBytes() const { return SlotHashes.capacity() * sizeof(SlotHash) + DirtySlots.capacity() * sizeof(uint64_t); }
    uint64_t ManifestBytes() const;
    // Checks trie structure and that all pinned Aliased nodes are reachable from the present.
    bool Check(std::string &why, const std::vector<Version> &all_versions) const;
    // Compare cached hashes with live data and report mismatching slot indices.
    // Call SettleHashes after the last write before checking.
    bool CheckHashes(std::string &why) const;

    enum class SlotState : uint8_t { Unhashed,
                                     Value,
                                     Default };
    struct SlotHash {
        Hash128 H{};
        SlotState State{};
        bool Dirty{};
    };

    Live L;
    const uint32_t Levels, BytesPerSlot;

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

    bool Busy{}; // Restore and LoadChanges assert against reentry while set
    bool ExternalWritesForbidden{}; // Assert on public Write calls during track restoration.
};
} // namespace store
