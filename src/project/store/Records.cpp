#include "project/store/Records.h"

#include <algorithm>
#include <bit>
#include <cassert>

namespace store {
std::span<const std::byte> Records::Read(uint64_t page) {
    assert(Present(page));
    const auto first = page * RecordsPerPage;
    const auto count = uint32_t(std::min<uint64_t>(RecordsPerPage, Length() - first));
    return EncodeRecordPage(count == RecordsPerPage ? UINT32_MAX : (1u << count) - 1u, PageScratch, [&](uint32_t slot) {
        C->Encode(Values, first + slot, Scratch);
        return std::span{Scratch};
    });
}

void Records::Write(uint64_t first, uint64_t count) {
    if (!count) return;
    assert(!Trie.ExternalWritesForbidden && "record write during track restoration");
    const auto page = first / RecordsPerPage, pages = (first + count - 1) / RecordsPerPage + 1 - page;
    Trie.Capture(page, pages, this, [](void *owner, uint64_t i) -> std::optional<Blob> {
        auto &records = *static_cast<Records *>(owner);
        return records.Present(i) ? std::optional{CopyBlob(records.Read(i))} : std::nullopt;
    });
    Trie.MarkDirty(page, pages);
}

void Records::Settle() {
    for (const auto i : Trie.Dirty()) Trie.Rehash(i, Present(i) ? std::optional{Read(i)} : std::nullopt);
    Trie.ClearDirty();
}

std::vector<uint32_t> Records::Changed() const {
    std::vector<uint32_t> changed;
    const auto length = std::max(Length(), ChangedLength);
    for (const auto page : Trie.ChangedSlots)
        for (uint64_t i = page * RecordsPerPage, end = std::min(length, (page + 1) * RecordsPerPage); i < end; ++i) changed.push_back(uint32_t(i));
    return changed;
}

std::vector<uint32_t> Records::TakeChanged() {
    auto changed = Changed();
    Trie.ChangedSlots.clear();
    ChangedLength = 0;
    return changed;
}

void Records::Apply(RestorePlan &plan) {
    const auto original = Length();
    if (Trie.CollectChanged) ChangedLength = std::max({ChangedLength, original, plan.Length});
    for (auto &c : plan.Changes) {
        const auto first = c.Slot * RecordsPerPage;
        const bool present = first < original;
        if ((c.Erase && !present) || (!c.Erase && present && c.MaybeEqual && Unchanged(Read(c.Slot), c.Incoming.View()))) {
            c.Unchanged = true;
            continue;
        }
        if (present) {
            c.Old = CopyBlob(Read(c.Slot));
            c.WasPresent = true;
        }
        if (c.Erase) {
            for (uint64_t i = first, end = std::min(original, first + RecordsPerPage); i < end; ++i) C->Reset(Values, i);
            continue;
        }
        auto bytes = c.Incoming.View();
        const auto mask = RecordMask(bytes);
        const auto count = uint32_t(std::popcount(mask));
        if (mask != (count == RecordsPerPage ? UINT32_MAX : (1u << count) - 1u)) throw std::invalid_argument("Non-contiguous record page.");
        if (first + count > Length()) C->Resize(Values, first + count);
        for (uint64_t i = first; i < first + count; ++i) C->Decode(Values, i, TakeRecord(bytes));
        if (!bytes.empty()) throw std::invalid_argument("Trailing record page bytes.");
        FreeBlob(c.Incoming);
    }
    C->Resize(Values, plan.Length);
}
} // namespace store
