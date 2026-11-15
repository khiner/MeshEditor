#pragma once

#include "Range.h"

#include <bit>
#include <deque>
#include <span>

namespace store {
struct History;
struct Records;
} // namespace store

// Store coalesced free ranges in address order.
// Cache subtree maximum lengths for first-fit lookup in at most 32 branches.
// History versions each node and the root.
struct RangeAllocator {
    static constexpr uint32_t Null = std::numeric_limits<uint32_t>::max();

    struct Node {
        uint32_t First{}, Last{}, Count{};
        uint32_t Left{Null}, Right{Null};
        bool Leaf() const { return Left == Null; }
        uint32_t Bit() const { return std::bit_width(First ^ Last) - 1u; }
    };
    struct State {
        uint32_t Root{Null}, Free{Null}, End{};
    };

    // Record changed allocator nodes.
    // Abandoning a transaction restores ownership without allocation.
    // Nested operations retain before-images in the enclosing transaction.
    class Transaction {
    public:
        explicit Transaction(RangeAllocator &owner)
            : Owner(owner), Parent(owner.Active), Before(owner.S), NodeCount(owner.Nodes.size()), JournalStart(owner.Journal.size()) { Owner.Active = this; }
        Transaction(const Transaction &) = delete;
        ~Transaction() {
            assert(Owner.Active == this);
            Owner.Active = Parent;
            if (Committed) {
                if (!Parent) Owner.Journal.clear();
                return;
            }
            for (auto i = Owner.Journal.size(); i-- > JournalStart;) Owner.Nodes[Owner.Journal[i].Index] = Owner.Journal[i].Before;
            Owner.Journal.resize(JournalStart);
            while (Owner.Nodes.size() > NodeCount) Owner.Nodes.pop_back();
            Owner.S = Before;
        }
        void Commit() { Committed = true; }

    private:
        friend struct RangeAllocator;
        RangeAllocator &Owner;
        Transaction *Parent;
        State Before;
        size_t NodeCount, JournalStart;
        bool Committed{};
    };

    // Versions each node and the root in the history under `name` while the returned records live.
    // A tracked allocator stays at its address.
    using TrackedRecords = std::unique_ptr<store::Records, void (*)(store::Records *)>;
    friend TrackedRecords TrackRecords(RangeAllocator &, store::History &, std::string name);

    Range Allocate(uint32_t count);
    void Free(Range range);

    // Free each contiguous run once, preserving the other live allocations.
    void Free(std::vector<Range>);

    // Failed fixed-address reservations do not dirty history.
    bool Reserve(Range range);

    uint32_t HighWaterMark() const { return S.End; }

    void Reset();

private:
    struct Change {
        uint32_t Index;
        Node Before;
    };

    State S;
    Transaction *Active{};
    // Retain before-images in mutation order.
    // Rollback discards nodes created after the enclosing transaction began.
    std::vector<Change> Journal;
    // Node addresses remain stable as the pool grows; freed nodes are reused.
    std::deque<Node> Nodes;
    store::Records *History{};

    // The root is record 0 and node n is record n + 1.
    uint64_t RecordCount() const { return uint64_t(Nodes.size()) + 1; }
    void ResizeRecords(uint64_t count) { Nodes.resize(count ? count - 1 : 0); }
    void EncodeRecord(uint64_t index, std::vector<std::byte> &out) const;
    void DecodeRecord(uint64_t index, std::span<const std::byte> bytes);
    void ResetRecord(uint64_t index) {
        if (index == 0) S = {};
        else Nodes[index - 1] = {};
    }

    State &WriteState();
    Node &Write(uint32_t n);
    uint32_t New(Node value);
    void Release(uint32_t n);
    void Refit(uint32_t n);
    uint32_t Predecessor(uint32_t key) const;
    uint32_t Successor(uint32_t key) const;
    void Replace(uint32_t parent, uint32_t from, uint32_t to);
    void Insert(uint32_t first, uint32_t count);
    void Remove(uint32_t first);
};
