#include "mesh/PageFootprint.h"

std::vector<uint32_t> RunBlocks(uint32_t first, uint32_t count) {
    std::vector<uint32_t> blocks;
    if (!count) return blocks;
    const auto last = uint32_t((uint64_t(first) + count - 1u) / MeshElementBlockSize);
    for (auto block = first / MeshElementBlockSize; block <= last; ++block) blocks.push_back(block);
    return blocks;
}

std::vector<uint32_t> WorkBlocks(const BufferArena<uint32_t> &arena, ElementWork work, uint32_t count, uint32_t origin) {
    if (work.Storage.Slot == InvalidSlot) return RunBlocks(origin, count);
    std::vector<uint32_t> blocks;
    ForEachWorkBlock(arena, work, [&](uint32_t block, auto) { blocks.push_back(block); });
    return blocks;
}

std::vector<uint32_t> PayloadBlocks(const BufferArena<uint32_t> &table, std::span<const uint32_t> blocks, bool required) {
    const auto entries = table.Buffer.GetSpan<uint32_t>();
    std::vector<uint32_t> payloads;
    payloads.reserve(blocks.size());
    for (const auto block : blocks) {
        const auto entry = block < entries.size() ? entries[block] : 0u;
        if (entry) payloads.push_back(entry - 1u);
        else if (required) throw std::out_of_range("A canonical write targets an absent attribute payload.");
    }
    return payloads;
}

void PageFootprint::Add(const mtl::Buffer &buffer, std::span<const uint32_t> blocks, uint64_t block_bytes) {
    if (blocks.empty()) return;
    auto found = std::ranges::find(Entries, &buffer, &Entry::Buffer);
    auto &pages = found == Entries.end() ? Entries.emplace_back(&buffer).Pages : found->Pages;
    for (const auto block : blocks) {
        const auto first = uint64_t(block) * block_bytes;
        for (auto page = uint32_t(first / mtl::HistoryPageBytes); page <= (first + block_bytes - 1u) / mtl::HistoryPageBytes; ++page)
            if (pages.empty() || pages.back() != page) pages.push_back(page);
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
    for (auto &entry : Entries) entry.Buffer->CaptureWritePages(Sort(entry));
}
