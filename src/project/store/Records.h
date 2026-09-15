#pragma once

#include "project/store/LiveTrie.h"

#include <cassert>
#include <zpp_bits.h>

namespace store {
// Versioned zpp-serialized records over an externally owned vector, or over one record.
// Write before mutating a record and Settle after writes complete.
struct Records {
    // Plain functions over the owner's storage, chosen once at construction.
    struct Codec {
        uint64_t (*Count)(const void *);
        void (*Resize)(void *, uint64_t);
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
    Records(void *values, const Codec &codec, uint32_t levels) : Values(values), C(&codec), Trie(levels) { Trie.MarkDirty(0, Length()); }
    Records(const Records &) = delete;
    Records &operator=(const Records &) = delete;

    uint64_t Length() const { return C->Count(Values); }
    bool Present(uint64_t i) const { return i < Length(); }
    // The returned span remains valid until the next Read or Encode.
    std::span<const std::byte> Read(uint64_t i) {
        Scratch.clear();
        C->Encode(Values, i, Scratch);
        return Scratch;
    }
    std::span<const std::byte> Encode(const Blob &value) const { return value.View(); }

    // Capture [first, first + count) before mutation.
    void Write(uint64_t first, uint64_t count) {
        assert(!Trie.ExternalWritesForbidden && "record write during track restoration");
        for (uint64_t i = first, last = first + count; i < last; ++i) {
            if (Trie.Uncaptured(i)) Trie.Capture(i, Present(i) ? std::optional{CopyBlob(Read(i))} : std::nullopt);
        }
        Trie.MarkDirty(first, count);
    }

    void Settle() {
        for (const auto i : Trie.Dirty()) Trie.Rehash(i, Present(i) ? std::optional{Read(i)} : std::nullopt);
        Trie.ClearDirty();
    }

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
    std::vector<std::byte> Scratch;
    LiveTrie Trie;

private:
    void Apply(RestorePlan &plan) {
        const auto original = Length();
        for (auto &c : plan.Changes) {
            const bool present = c.Slot < original;
            if (c.Erase) {
                if (!present) {
                    c.Unchanged = true;
                    continue;
                }
                c.Old = CopyBlob(Read(c.Slot));
                c.WasPresent = true;
                C->Reset(Values, c.Slot);
                continue;
            }
            if (present && c.MaybeEqual && Unchanged(Read(c.Slot), c.Incoming.View())) {
                c.Unchanged = true;
                continue;
            }
            if (present) {
                c.Old = CopyBlob(Read(c.Slot));
                c.WasPresent = true;
            } else if (c.Slot >= Length()) C->Resize(Values, c.Slot + 1);
            C->Decode(Values, c.Slot, c.Incoming.View());
            FreeBlob(c.Incoming);
        }
        C->Resize(Values, plan.Length);
    }
};
} // namespace store
