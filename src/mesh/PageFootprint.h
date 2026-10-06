#pragma once

#include "gpu/ElementWork.h"
#include "mesh/ElementAttribute.h"
#include "metal/Buffer.h"
#include "metal/BufferArena.h"
#include "render/ElementWorkOps.h"

// The bytes of one element block of T records.
template<typename T> inline constexpr uint64_t BlockBytes = uint64_t(MeshElementBlockSize) * sizeof(T);

// Appends the element block of a handle unless it repeats the last block.
inline void AddBlock(std::vector<uint32_t> &blocks, uint32_t handle) {
    const auto block = handle / MeshElementBlockSize;
    if (blocks.empty() || blocks.back() != block) blocks.push_back(block);
}
// Ascending element blocks of the handle run [first, first + count).
std::vector<uint32_t> RunBlocks(uint32_t first, uint32_t count);
// Ascending element blocks of sparse work, or of the `count` handles from `origin` when the work has no storage.
std::vector<uint32_t> WorkBlocks(const BufferArena<uint32_t> &, ElementWork, uint32_t count, uint32_t origin = 0u);
// Visits the ascending handles of sparse work, or the `count` handles from `origin` when the work has no storage.
void ForEachWorkHandle(const BufferArena<uint32_t> &arena, ElementWork work, uint32_t count, uint32_t origin, auto &&visit) {
    if (work.Storage.Slot != InvalidSlot) ForEachWorkElement(arena, work, visit);
    else
        for (uint32_t i = 0u; i < count; ++i) visit(origin + i);
}
// The payload blocks a table of payload block plus one names for element blocks.
// A zero or missing entry has no payload, which throws when the payload is required.
std::vector<uint32_t> PayloadBlocks(const BufferArena<uint32_t> &table, std::span<const uint32_t> blocks, bool required = false);

// Sorted unique history pages of each canonical buffer an edit reads or writes.
// Element block b of a buffer covers bytes [b * block_bytes, (b + 1) * block_bytes).
class PageFootprint {
public:
    void Add(const mtl::Buffer &, std::span<const uint32_t> blocks, uint64_t block_bytes);
    // The table entries of element blocks and the payload blocks they name.
    template<typename T> void Attribute(const ElementAttribute<T> &attribute, std::span<const uint32_t> blocks, uint32_t entries = 1u) {
        Add(attribute.Blocks.Buffer, blocks, sizeof(uint32_t));
        AttributeValues(attribute, blocks, false, entries);
    }
    // Every entry's payload block of each element block, where a required payload throws when absent.
    template<typename T> void AttributeValues(const ElementAttribute<T> &attribute, std::span<const uint32_t> blocks, bool required, uint32_t entries = 1u) {
        std::vector<uint32_t> payloads;
        for (const auto first : PayloadBlocks(attribute.Blocks, blocks, required))
            for (uint32_t entry = 0u; entry < entries; ++entry) payloads.push_back(first + entry);
        Add(attribute.Values.Buffer, payloads, sizeof(typename ElementAttribute<T>::Block));
    }

    // The buffer's history pages, empty when nothing was added for it.
    std::span<const uint32_t> Pages(const mtl::Buffer &);
    // Captures each buffer's pages before the GPU writes them.
    void CaptureWrites();

private:
    struct Entry {
        const mtl::Buffer *Buffer;
        std::vector<uint32_t> Pages;
    };
    std::vector<Entry> Entries;
    static std::span<const uint32_t> Sort(Entry &);
};
