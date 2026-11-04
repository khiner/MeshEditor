#pragma once

#include "gpu/MeshElementBlock.h"
#include "render/ElementWorkOps.h"

template<typename T> inline constexpr uint64_t BlockBytes = uint64_t(MeshElementBlockSize) * sizeof(T);

class IndexMarks {
public:
    // Returns true when the index is newly marked.
    bool Mark(uint32_t index) {
        if (index / 32u >= Words.size()) Words.resize(index / 32u + 1u);
        auto &word = Words[index / 32u];
        const bool fresh = !(word >> (index % 32u) & 1u);
        word |= 1u << (index % 32u);
        return fresh;
    }
    std::vector<uint32_t> Ascending() const { return SetBitHandles(Words); }

private:
    std::vector<uint32_t> Words;
};

// Appends the element block of an arena address unless it repeats the last block.
inline void AddBlock(std::vector<uint32_t> &blocks, uint32_t address) {
    const auto block = address / MeshElementBlockSize;
    if (blocks.empty() || blocks.back() != block) blocks.push_back(block);
}
std::vector<uint32_t> TableBlocks(std::span<const uint32_t> table, std::span<const uint32_t> locals);
std::vector<uint32_t> RunBlocks(std::span<const uint32_t> table, uint32_t first, uint32_t count);
// The arena blocks of sparse work's handles, or of the first `count` handles when the work has no storage, through the record's table entries.
std::vector<uint32_t> WorkBlocks(const BufferArena<uint32_t> &, ElementWork, uint32_t count, std::span<const uint32_t> table);
// Visits the ascending handles of sparse work, or the `count` handles from `origin` when the work has no storage.
void ForEachWorkHandle(const BufferArena<uint32_t> &arena, ElementWork work, uint32_t count, uint32_t origin, auto &&visit) {
    if (work.Storage.Slot != InvalidSlot) ForEachWorkElement(arena, work, visit);
    else {
        for (uint32_t i = 0u; i < count; ++i) visit(origin + i);
    }
}

// Sorted unique history pages of each canonical buffer an edit reads or writes.
class PageFootprint {
public:
    void Add(const mtl::Buffer &, std::span<const uint32_t> blocks, uint64_t block_bytes);
    // Adds every entry's values of a record's layer range at the local blocks `members` holds for the arena blocks, and nothing for an empty range.
    template<typename Block> void Layer(const BufferArena<Block> &values, Range range, std::span<const MeshElementBlock> members, std::span<const uint32_t> blocks, uint32_t entries = 1u) {
        if (!range.Count) return;
        std::vector<uint32_t> layer;
        layer.reserve(blocks.size() * entries);
        for (const auto block : blocks) {
            for (uint32_t entry = 0u; entry < entries; ++entry) layer.push_back(range.Offset + members[block].Local * entries + entry);
        }
        Add(values.Buffer, layer, sizeof(Block));
    }

    std::span<const uint32_t> Pages(const mtl::Buffer &);
    // Captures each buffer's pages before the GPU writes them.
    void CaptureWrites();

private:
    struct Entry {
        const mtl::Buffer *Buffer;
        IndexMarks Marks;
        std::vector<uint32_t> Pages;
    };
    std::vector<Entry> Entries;
    static std::span<const uint32_t> Collect(Entry &);
};
