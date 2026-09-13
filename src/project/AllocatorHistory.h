#pragma once

#include "RangeAllocator.h"
#include "project/VectorHistory.h"

namespace project {
inline store::Live AllocatorEndSlots(uint32_t &value) {
    store::Live live;
    live.Length = [] { return 1; };
    live.Present = [](uint64_t i) { return i == 0; };
    live.Read = [&value](uint64_t) { return std::as_bytes(std::span{&value, 1}); };
    live.Replace = [&value](uint64_t, store::Blob incoming, bool &was_present) {
        const auto old = value;
        std::memcpy(&value, incoming.Data, sizeof(value));
        std::memcpy(incoming.Data, &old, sizeof(value));
        was_present = true;
        return incoming;
    };
    live.Erase = [&value](uint64_t) {
        const auto old = store::CopyBlob(std::as_bytes(std::span{&value, 1}));
        value = {};
        return old;
    };
    return live;
}
struct AllocatorHistory {
    AllocatorHistory(RangeAllocator &allocator, store::History &history, const std::string &name)
        : Allocator(allocator), Free(allocator.FreeBlocks, history, name + ".free"), End(AllocatorEndSlots(allocator.EndOffset), 1, sizeof(allocator.EndOffset)) {
        Allocator.FreeHistory = &Free.Trie;
        Allocator.EndHistory = &End;
        history.Track(End, name + ".end", 0);
        End.Write(0, 1);
    }
    ~AllocatorHistory() {
        Allocator.FreeHistory = nullptr;
        Allocator.EndHistory = nullptr;
    }

    RangeAllocator &Allocator;
    VectorHistory<Range> Free;
    store::LiveTrie End;
};
} // namespace project
