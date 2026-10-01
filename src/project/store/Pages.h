#pragma once

#include "project/store/LiveTrie.h"

#include <algorithm>
#include <cassert>
#include <cstring>

namespace store {
// Versioned zero-default byte pages over CPU or Metal storage.
// Storage holds whole pages and grows through Reserve, which returns the new storage.
// The backing owns the live length.
// Write before mutating live bytes and Settle after CPU and GPU writes complete.
struct Pages {
    using ReserveFn = std::span<std::byte> (*)(void *backing, uint64_t bytes);

    // Pages over an owned byte vector.
    explicit Pages(uint32_t page_bytes = 512, uint32_t levels = 5)
        : PageBytes(page_bytes), Len(&OwnedLen), Trie(levels, page_bytes), Backing(&Bytes), Reserve(&ReserveBytes) {}
    // Pages over external storage.
    Pages(uint32_t page_bytes, uint32_t levels, void *backing, ReserveFn reserve, uint64_t &length)
        : PageBytes(page_bytes), Len(&length), Trie(levels, page_bytes), Backing(backing), Reserve(reserve) {}
    Pages(const Pages &) = delete;
    Pages &operator=(const Pages &) = delete;

    uint64_t Length() const { return *Len; }
    const std::byte *Data() const { return Storage.data(); }
    bool Present(uint64_t page) const { return (page + 1) * PageBytes <= Storage.size(); }
    std::span<const std::byte> Read(uint64_t page) const { return Storage.subspan(page * PageBytes, PageBytes); }
    std::span<const std::byte> Encode(const Blob &value) const { return value.View(); }

    // Grow storage to cover at least bytes.
    void Grow(uint64_t bytes) {
        const auto needed = Trie.SlotsFor(bytes) * PageBytes;
        if (needed > Storage.size()) Storage = Reserve(Backing, needed);
    }
    // Capture old bytes and grow storage to cover the range.
    void Write(uint64_t offset, uint64_t size) {
        if (size == 0) return;
        assert(!Trie.ExternalWritesForbidden && "page write during track restoration");
        Grow(offset + size);
        const auto first = offset / PageBytes, last = (offset + size - 1) / PageBytes;
        Trie.Write(first, last - first + 1, Storage);
    }
    // Capture old bytes and return a writable span.
    std::span<std::byte> Mutable(uint64_t offset, uint64_t size) {
        Write(offset, size);
        return Storage.subspan(offset, size);
    }
    // Capture removed bytes and zero storage beyond the new length.
    void Resize(uint64_t len) {
        const auto current = *Len;
        if (len > current) {
            Write(current, len - current);
        } else if (len < current) {
            Write(len, current - len);
            std::memset(Storage.data() + len, 0, current - len);
        }
        *Len = len;
    }

    void Settle() { Trie.SettleFlat(Storage); }

    // The byte ranges a restore or load changed, for an owner that maps bytes to records. The consumer takes them.
    std::vector<std::pair<uint64_t, uint64_t>> TakeChangedExtents() { return std::exchange(ChangedExtents, {}); }

    bool Restore(const Version &v) {
        Settle();
        auto plan = Trie.PlanRestore(v);
        Apply(plan);
        *Len = plan.Length;
        Grow(plan.Length);
        return Trie.CommitRestore(std::move(plan));
    }
    void Load(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves) {
        Settle();
        auto plan = Trie.PlanLoad(length, changes, leaves);
        Apply(plan);
        *Len = plan.Length;
        Grow(plan.Length);
        Trie.CommitLoad(std::move(plan));
    }

    uint32_t PageBytes;
    std::span<std::byte> Storage;
    uint64_t *Len; // The live length, owned by the backing.
    LiveTrie Trie;
    std::vector<std::byte> Bytes;
    uint64_t OwnedLen{}; // The length of Bytes-backed pages.
    void *Backing;
    ReserveFn Reserve;

private:
    std::vector<std::pair<uint64_t, uint64_t>> ChangedExtents;

    static std::span<std::byte> ReserveBytes(void *backing, uint64_t bytes) {
        auto &vector = *static_cast<std::vector<std::byte> *>(backing);
        vector.resize(bytes);
        return vector;
    }
    // The first and one past the last byte where two equal-size pages differ, or an empty extent when they are equal.
    static std::pair<uint32_t, uint32_t> Difference(std::span<const std::byte> a, std::span<const std::byte> b) {
        constexpr uint32_t Block = 256;
        const auto size = uint32_t(a.size());
        const auto differs = [&](uint32_t at) {
            uint64_t bits = 0;
            for (uint32_t i = at; i < at + Block; i += sizeof(uint64_t)) {
                uint64_t x, y;
                std::memcpy(&x, a.data() + i, sizeof(x));
                std::memcpy(&y, b.data() + i, sizeof(y));
                bits |= x ^ y;
            }
            return bits != 0;
        };
        uint32_t first = 0;
        while (first + Block <= size && !differs(first)) first += Block;
        while (first < size && a[first] == b[first]) ++first;
        if (first == size) return {};
        uint32_t end = size;
        while (end - first >= Block && !differs(end - Block)) end -= Block;
        while (a[end - 1] == b[end - 1]) --end;
        return {first, end};
    }
    void NoteChanged(uint64_t slot, std::pair<uint32_t, uint32_t> extent) {
        ChangedExtents.emplace_back(slot * PageBytes + extent.first, slot * PageBytes + extent.second);
    }
    void Apply(RestorePlan &plan) {
        for (auto &c : plan.Changes) {
            if (c.Erase) {
                // Settled hashes identify the present pages that hold only zeros.
                if (!Present(c.Slot) || c.Slot >= Trie.SlotHashes.size() || Trie.SlotHashes[c.Slot].State != LiveTrie::SlotState::Value) {
                    c.Unchanged = true;
                    continue;
                }
                const auto page = Storage.subspan(c.Slot * PageBytes, PageBytes);
                c.Old = CopyBlob(page);
                c.WasPresent = true;
                std::memset(page.data(), 0, PageBytes);
                NoteChanged(c.Slot, Difference(c.Old.View(), page));
                continue;
            }
            Grow((c.Slot + 1) * PageBytes);
            const auto page = Storage.subspan(c.Slot * PageBytes, PageBytes);
            assert(c.Incoming.Size == PageBytes);
            const auto extent = Difference(page, c.Incoming.View());
            if (extent.first == extent.second) {
                c.Unchanged = true;
                continue;
            }
            // The pages agree outside the extent, so exchanging it exchanges the whole pages.
            std::swap_ranges(page.begin() + extent.first, page.begin() + extent.second, c.Incoming.Data + extent.first);
            c.Old = std::exchange(c.Incoming, {});
            c.WasPresent = true;
            NoteChanged(c.Slot, extent);
        }
    }
};

// A typed view over pages of trivially copyable records.
template<typename T>
struct VersionedVector {
    static_assert(std::is_trivially_copyable_v<T>);

    explicit VersionedVector(uint32_t page_bytes = 512) : P(page_bytes) {}

    size_t size() const { return P.Length() / sizeof(T); }
    bool empty() const { return size() == 0; }
    std::span<const T> View() const { return {reinterpret_cast<const T *>(P.Data()), size()}; }
    const T &operator[](size_t i) const { return View()[i]; }

    T &Mutable(size_t i) {
        assert(i < size());
        return *reinterpret_cast<T *>(P.Mutable(i * sizeof(T), sizeof(T)).data());
    }
    void Set(size_t i, const T &v) { Mutable(i) = v; }
    void PushBack(const T &v) {
        const auto i = size();
        P.Resize((i + 1) * sizeof(T));
        std::memcpy(P.Storage.data() + i * sizeof(T), &v, sizeof(T));
    }
    void PopBack() { P.Resize((size() - 1) * sizeof(T)); }
    T Back() const { return (*this)[size() - 1]; }
    void Clear() { P.Resize(0); }

    Pages P;
};
} // namespace store
