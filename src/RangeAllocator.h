#pragma once

#include "Range.h"
#include "project/store/Records.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <deque>
#include <limits>
#include <stdexcept>
#include <vector>

// Coalesced free ranges in a compressed binary radix tree, ordered by address.
// A subtree's maximum free length finds the first fitting range in at most 32
// branches. History versions individual nodes and the root state.
struct RangeAllocator {
    static constexpr uint32_t Null = std::numeric_limits<uint32_t>::max();
    static constexpr uint32_t HistoryLevels = 7;

    struct Node {
        uint32_t First{}, Last{}, Count{};
        uint32_t Left{Null}, Right{Null};
        bool Leaf() const { return Left == Null; }
        uint32_t Bit() const { return std::bit_width(First ^ Last) - 1u; }
    };
    struct State {
        uint32_t Root{Null}, Free{Null}, End{};
    };

    // A reservation journals only changed allocator nodes. Abandoning it
    // restores ownership without allocating memory or copying the arena.
    // Nested operations retain their before-images in the enclosing scope.
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

    store::Records *History{};

    Range Allocate(uint32_t count) {
        if (count == 0) return {};
        Transaction transaction{*this};
        auto n = S.Root;
        if (n == Null || Nodes[n].Count < count) {
            if (count > Null - S.End) throw std::length_error("Arena index space exhausted.");
            const auto first = S.End;
            WriteState().End += count;
            transaction.Commit();
            return {first, count};
        }
        while (!Nodes[n].Leaf()) n = Nodes[Nodes[n].Left].Count >= count ? Nodes[n].Left : Nodes[n].Right;
        const auto first = Nodes[n].First, available = Nodes[n].Count;
        Remove(first);
        if (available > count) Insert(first + count, available - count);
        transaction.Commit();
        return {first, count};
    }

    void Free(Range range) {
        if (range.Count == 0) return;
        if (range.Offset > S.End || range.Count > S.End - range.Offset) throw std::out_of_range("Free range exceeds its arena.");
        auto first = range.Offset, end = first + range.Count;
        const auto before = Predecessor(first), after = Successor(first);
        if ((before != Null && Nodes[before].First + Nodes[before].Count > first) ||
            (after != Null && Nodes[after].First < end))
            throw std::logic_error("Arena range freed twice or overlaps free storage.");
        Transaction transaction{*this};
        if (before != Null && Nodes[before].First + Nodes[before].Count == first) {
            first = Nodes[before].First;
            Remove(first);
        }
        const auto next = Successor(end);
        if (next != Null && Nodes[next].First == end) {
            const auto next_first = Nodes[next].First;
            end += Nodes[next].Count;
            Remove(next_first);
        }
        Insert(first, end - first);
        transaction.Commit();
    }

    // Free each contiguous run once, preserving the other live allocations.
    void Free(std::vector<Range>);

    // Failed fixed-address reservations do not dirty history.
    bool Reserve(Range range) {
        if (range.Count == 0) return true;
        if (range.Count > Null - range.Offset) return false;
        const auto end = range.Offset + range.Count;
        Transaction transaction{*this};
        if (range.Offset >= S.End) {
            const auto previous = S.End;
            WriteState().End = end;
            if (range.Offset > previous) Free({previous, range.Offset - previous});
            transaction.Commit();
            return true;
        }
        const auto before = Predecessor(range.Offset);
        if (before == Null) return false;
        const auto first = Nodes[before].First, block_end = first + Nodes[before].Count;
        if (end > block_end) return false;
        Remove(first);
        if (first < range.Offset) Insert(first, range.Offset - first);
        if (end < block_end) Insert(end, block_end - end);
        transaction.Commit();
        return true;
    }

    uint32_t HighWaterMark() const { return S.End; }

    void Reset() {
        if (Active) throw std::logic_error("Cannot reset an arena with pending reservations.");
        if (History) History->Write(0, RecordCount());
        S = {};
        Nodes.clear();
    }

    uint64_t RecordCount() const { return uint64_t(Nodes.size()) + 1; }
    void ResizeRecords(uint64_t count) { Nodes.resize(count ? count - 1 : 0); }
    void EncodeRecord(uint64_t index, std::vector<std::byte> &out) const {
        zpp::bits::out archive{out};
        if (index == 0) archive(S).or_throw();
        else archive(Nodes[index - 1]).or_throw();
        out.resize(archive.position());
    }
    void DecodeRecord(uint64_t index, std::span<const std::byte> bytes) {
        if (index == 0) zpp::bits::in{bytes}(S).or_throw();
        else zpp::bits::in{bytes}(Nodes[index - 1]).or_throw();
    }
    void ResetRecord(uint64_t index) {
        if (index == 0) S = {};
        else Nodes[index - 1] = {};
    }

private:
    struct Change { uint32_t Index; Node Before; };

    State S;
    Transaction *Active{};
    // Before-images of the nodes the active transactions changed, oldest first.
    // The innermost transaction's node count bounds the captured nodes, since every enclosing one discards the nodes created after it began.
    std::vector<Change> Journal;
    // Node addresses remain stable as the pool grows. Freed nodes are reused.
    std::deque<Node> Nodes;

    State &WriteState() {
        if (History) History->Write(0, 1);
        return S;
    }
    Node &Write(uint32_t n) {
        if (Active && n < Active->NodeCount) Journal.push_back({n, Nodes[n]});
        if (History) History->Write(uint64_t(n) + 1, 1);
        return Nodes[n];
    }
    uint32_t New(Node value) {
        if (S.Free != Null) {
            const auto n = S.Free;
            WriteState().Free = Nodes[n].Left;
            Write(n) = value;
            return n;
        }
        if (Nodes.size() >= Null) throw std::length_error("Arena allocator node space exhausted.");
        const auto n = uint32_t(Nodes.size());
        if (History) History->Write(uint64_t(n) + 1, 1);
        Nodes.push_back(value);
        return n;
    }
    void Release(uint32_t n) {
        Write(n) = {.Left = S.Free};
        WriteState().Free = n;
    }
    void Refit(uint32_t n) {
        const auto &a = Nodes[Nodes[n].Left], &b = Nodes[Nodes[n].Right];
        auto &node = Write(n);
        node.First = a.First;
        node.Last = b.Last;
        node.Count = std::max(a.Count, b.Count);
    }
    uint32_t Predecessor(uint32_t key) const {
        auto n = S.Root;
        if (n == Null || Nodes[n].First > key) return Null;
        while (!Nodes[n].Leaf()) n = Nodes[Nodes[n].Right].First <= key ? Nodes[n].Right : Nodes[n].Left;
        return n;
    }
    uint32_t Successor(uint32_t key) const {
        auto n = S.Root;
        if (n == Null || Nodes[n].Last < key) return Null;
        while (!Nodes[n].Leaf()) n = Nodes[Nodes[n].Left].Last >= key ? Nodes[n].Left : Nodes[n].Right;
        return n;
    }
    void Replace(uint32_t parent, uint32_t from, uint32_t to) {
        if (parent == Null) WriteState().Root = to;
        else {
            auto &node = Write(parent);
            (node.Left == from ? node.Left : node.Right) = to;
        }
    }
    void Insert(uint32_t first, uint32_t count) {
        if (S.Root == Null) {
            const auto n = New({first, first, count});
            WriteState().Root = n;
            return;
        }
        auto leaf = S.Root;
        while (!Nodes[leaf].Leaf()) leaf = ((first >> Nodes[leaf].Bit()) & 1u) ? Nodes[leaf].Right : Nodes[leaf].Left;
        assert(first != Nodes[leaf].First);
        const auto bit = std::bit_width(first ^ Nodes[leaf].First) - 1u;
        std::array<uint32_t, 32> path;
        uint32_t depth = 0, n = S.Root;
        while (!Nodes[n].Leaf() && Nodes[n].Bit() > bit) {
            path[depth++] = n;
            n = ((first >> Nodes[n].Bit()) & 1u) ? Nodes[n].Right : Nodes[n].Left;
        }
        const auto added = New({first, first, count});
        const auto left = ((first >> bit) & 1u) ? n : added;
        const auto right = left == n ? added : n;
        const auto branch = New({Nodes[left].First, Nodes[right].Last, std::max(Nodes[left].Count, Nodes[right].Count), left, right});
        Replace(depth ? path[depth - 1] : Null, n, branch);
        while (depth) Refit(path[--depth]);
    }
    void Remove(uint32_t first) {
        std::array<uint32_t, 32> path;
        uint32_t depth = 0, n = S.Root;
        while (!Nodes[n].Leaf()) {
            path[depth++] = n;
            n = ((first >> Nodes[n].Bit()) & 1u) ? Nodes[n].Right : Nodes[n].Left;
        }
        assert(Nodes[n].First == first);
        if (!depth) Replace(Null, n, Null);
        else {
            const auto parent = path[--depth];
            const auto sibling = Nodes[parent].Left == n ? Nodes[parent].Right : Nodes[parent].Left;
            Replace(depth ? path[depth - 1] : Null, parent, sibling);
            Release(parent);
        }
        Release(n);
        while (depth) Refit(path[--depth]);
    }
};

inline constexpr store::Records::Codec AllocatorCodec{
    [](const void *v) { return static_cast<const RangeAllocator *>(v)->RecordCount(); },
    [](void *v, uint64_t n) { static_cast<RangeAllocator *>(v)->ResizeRecords(n); },
    [](const void *v, uint64_t i, std::vector<std::byte> &out) { static_cast<const RangeAllocator *>(v)->EncodeRecord(i, out); },
    [](void *v, uint64_t i, std::span<const std::byte> bytes) { static_cast<RangeAllocator *>(v)->DecodeRecord(i, bytes); },
    [](void *v, uint64_t i) { static_cast<RangeAllocator *>(v)->ResetRecord(i); },
};
