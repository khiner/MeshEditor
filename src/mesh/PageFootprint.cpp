#include "mesh/PageFootprint.h"

std::vector<uint32_t> TableBlocks(std::span<const uint32_t> table, std::span<const uint32_t> locals) {
    std::vector<uint32_t> blocks;
    blocks.reserve(locals.size());
    for (const auto local : locals) {
        if (local < table.size() && table[local] != InvalidOffset) blocks.push_back(table[local]);
    }
    return blocks;
}

std::vector<uint32_t> RunBlocks(std::span<const uint32_t> table, uint32_t first, uint32_t count) {
    std::vector<uint32_t> locals;
    if (!count) return locals;
    const auto last = uint32_t((uint64_t(first) + count - 1u) / MeshElementBlockSize);
    for (auto block = first / MeshElementBlockSize; block <= last; ++block) locals.push_back(block);
    return TableBlocks(table, locals);
}

std::vector<uint32_t> WorkBlocks(const BufferArena<uint32_t> &arena, ElementWork work, uint32_t count, std::span<const uint32_t> table) {
    if (work.Storage.Slot == InvalidSlot) return RunBlocks(table, 0u, count);
    std::vector<uint32_t> locals;
    ForEachWorkBlock(arena, work, [&](uint32_t block, auto) { locals.push_back(block); });
    return TableBlocks(table, locals);
}

void PageFootprint::Add(const mtl::Buffer &buffer, std::span<const uint32_t> blocks, uint64_t block_bytes) {
    if (blocks.empty()) return;
    const auto found = std::ranges::find(Entries, &buffer, &Entry::Buffer);
    auto &entry = found == Entries.end() ? Entries.emplace_back(&buffer) : *found;
    auto &pages = entry.Pages;
    for (const auto block : blocks) {
        const auto first = uint64_t(block) * block_bytes;
        if (buffer.Writes) entry.Extents.emplace_back(first, first + block_bytes);
        for (auto page = uint32_t(first / mtl::HistoryPageBytes); page <= (first + block_bytes - 1u) / mtl::HistoryPageBytes; ++page) {
            if (pages.empty() || pages.back() != page) pages.push_back(page);
        }
    }
}

std::span<const uint32_t> PageFootprint::Sort(Entry &entry) {
    std::ranges::sort(entry.Pages);
    entry.Pages.erase(std::unique(entry.Pages.begin(), entry.Pages.end()), entry.Pages.end());
    return entry.Pages;
}

std::span<const uint32_t> PageFootprint::Pages(const mtl::Buffer &buffer) {
    const auto found = std::ranges::find(Entries, &buffer, &Entry::Buffer);
    return found == Entries.end() ? std::span<const uint32_t>{} : Sort(*found);
}

void PageFootprint::CaptureWrites() {
    for (auto &entry : Entries) {
        for (const auto [begin, end] : entry.Extents) entry.Buffer->LogWrite(begin, end);
        entry.Buffer->CaptureWritePages(Sort(entry));
    }
}
