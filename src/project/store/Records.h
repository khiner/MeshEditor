#pragma once

#include "project/store/LiveTrie.h"
#include "project/store/RecordPage.h"

#include <zpp_bits.h>

namespace store {
// Versioned pages of serialized records over externally owned storage.
// Write before mutating a record and Settle after writes complete.
struct Records {
    // Plain functions over the owner's storage, chosen once at construction.
    struct Codec {
        uint64_t (*Count)(const void *);
        void (*Resize)(void *, uint64_t);
        // Replace out's contents; the scratch storage is reused between records.
        void (*Encode)(const void *, uint64_t index, std::vector<std::byte> &out);
        void (*Decode)(void *, uint64_t index, std::span<const std::byte>);
        void (*Reset)(void *, uint64_t index);
    };
    template<typename T> static constexpr Codec VectorCodec{
        [](const void *v) { return uint64_t(static_cast<const std::vector<T> *>(v)->size()); },
        [](void *v, uint64_t n) { static_cast<std::vector<T> *>(v)->resize(n); },
        [](const void *v, uint64_t i, std::vector<std::byte> &out) {
            zpp::bits::out archive{out};
            // zpp reflects large aggregates through non-const references.
            archive(const_cast<T &>((*static_cast<const std::vector<T> *>(v))[i])).or_throw();
            out.resize(archive.position());
        },
        [](void *v, uint64_t i, std::span<const std::byte> bytes) { zpp::bits::in{bytes}((*static_cast<std::vector<T> *>(v))[i]).or_throw(); },
        [](void *v, uint64_t i) { (*static_cast<std::vector<T> *>(v))[i] = {}; },
    };

    template<typename T>
    explicit Records(std::vector<T> &values, uint32_t levels = 4) : Records(&values, VectorCodec<T>, levels) {}
    Records(void *values, const Codec &codec, uint32_t levels) : Values(values), C(&codec), Trie(levels, 0, RecordsPerPage) { Trie.MarkDirty(0, Trie.SlotsFor(Length())); }
    Records(const Records &) = delete;
    Records &operator=(const Records &) = delete;

    uint64_t Length() const { return C->Count(Values); }
    bool Present(uint64_t page) const { return page * RecordsPerPage < Length(); }
    // Read an encoded page. The span remains valid until the next Read.
    std::span<const std::byte> Read(uint64_t page);
    std::span<const std::byte> Encode(const Blob &value) const { return value.View(); }

    // Capture [first, first + count) before mutation.
    void Write(uint64_t first, uint64_t count);

    void Settle();

    bool Restore(const Version &v) {
        Settle();
        auto plan = Trie.PlanRestore(v);
        Apply(plan);
        return Trie.CommitRestore(std::move(plan));
    }
    void Load(uint64_t length, std::span<const std::pair<uint64_t, Hash128>> changes, const std::unordered_map<Hash128, std::vector<std::byte>, Hash128Hasher> &leaves) {
        Settle();
        auto plan = Trie.PlanLoad(length, changes, leaves);
        Apply(plan);
        Trie.CommitLoad(std::move(plan));
    }

    void *Values;
    const Codec *C;
    std::vector<std::byte> Scratch, PageScratch;
    LiveTrie Trie;

    std::vector<uint32_t> Changed() const;
    std::vector<uint32_t> TakeChanged();

private:
    uint64_t ChangedLength{};
    void Apply(RestorePlan &plan);
};
} // namespace store
