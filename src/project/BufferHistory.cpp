#include "project/BufferHistory.h"
#include "metal/Buffer.h"

#include <algorithm>

namespace project {
namespace {
std::span<std::byte> Page(const BufferHistory &history, uint64_t page) {
    return history.B->Contents().subspan(page * history.PageBytes, history.PageBytes);
}
store::Live BufferSlots(BufferHistory &self) {
    store::Live live;
    live.PageBytes = self.PageBytes;
    live.Length = [&self] { return self.B->UsedSize; };
    live.SetLength = [&self](uint64_t length) {
        self.B->Reserve(store::SlotsFor(self.Trie.L, length) * self.PageBytes);
        self.B->UsedSize = length;
    };
    live.Present = [&self](uint64_t page) { return (page + 1) * self.PageBytes <= self.B->Contents().size(); };
    live.Read = [&self](uint64_t page) -> std::span<const std::byte> { return Page(self, page); };
    live.Replace = [&self](uint64_t page, store::Blob incoming, bool &was_present) {
        self.B->Reserve((page + 1) * self.PageBytes);
        was_present = true;
        return store::SwapBlob(Page(self, page), incoming);
    };
    live.Erase = [&self](uint64_t page) {
        const auto old = store::CopyBlob(Page(self, page));
        std::ranges::fill(Page(self, page), std::byte{});
        return old;
    };
    return live;
}
} // namespace

BufferHistory::BufferHistory(mtl::Buffer &buffer, store::History &history, std::string name, uint32_t page_bytes)
    : B(&buffer), PageBytes(page_bytes), Trie(BufferSlots(*this), 5, page_bytes) {
    B->Reserve(store::SlotsFor(Trie.L, B->UsedSize) * PageBytes);
    const auto bytes = B->Contents();
    if (bytes.size() > B->UsedSize) std::ranges::fill(bytes.subspan(B->UsedSize), std::byte{});
    Trie.CollectChanged = true;
    history.Track(Trie, std::move(name), 0);
    if (B->UsedSize) Trie.Write(0, store::SlotsFor(Trie.L, B->UsedSize));
}

void BufferHistory::Write(uint64_t offset, uint64_t size) {
    if (!size) return;
    const auto first = offset / PageBytes;
    const auto end = store::SlotsFor(Trie.L, offset + size);
    B->Reserve(end * PageBytes);
    Trie.Write(first, end - first);
}
} // namespace project
