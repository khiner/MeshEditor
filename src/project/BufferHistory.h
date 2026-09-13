#pragma once

#include "project/store/History.h"

namespace mtl {
struct Buffer;
}

namespace project {
// Track mapped Metal pages in place and copy changed pages for pinned versions.
struct BufferHistory {
    mtl::Buffer *B;
    uint32_t PageBytes;
    std::vector<std::byte> Scratch;
    store::LiveTrie Trie;

    BufferHistory(mtl::Buffer &, store::History &, std::string name, uint32_t page_bytes);
    void Write(uint64_t offset, uint64_t size);
};
} // namespace project
