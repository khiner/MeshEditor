#pragma once

#include "Range.h"

#include <algorithm>
#include <vector>

// Elements one threadgroup covers outside the scan passes, matching the shaders' ScanTileSize.
constexpr uint32_t TileElements{256};
// Elements one scan block covers, four per thread, matching the shaders' ScanBlockElements.
constexpr uint32_t BlockElements{1024};

constexpr uint32_t TileCount(uint32_t count, uint32_t per_tile) { return (count + per_tile - 1) / per_tile; }

// Runs of a work list, each sized so one submit's scratch stays under the budget.
// An item wider than the budget takes a chunk to itself.
struct ScratchChunks {
    std::vector<Range> Chunks;
    uint32_t WidestWords{}; // Scratch words the widest chunk takes
    uint32_t MostJobs{}; // Items the fullest chunk holds
};

// Split `count` items into budgeted chunks, `words_of(i)` giving item i's scratch demand.
// The widest and fullest counts size buffers every chunk writes over.
ScratchChunks ChunkByScratch(uint32_t count, uint32_t budget_words, auto &&words_of) {
    ScratchChunks split;
    uint32_t first = 0, words = 0;
    const auto close = [&](uint32_t end) {
        split.Chunks.emplace_back(first, end - first);
        split.WidestWords = std::max(split.WidestWords, words);
        split.MostJobs = std::max(split.MostJobs, end - first);
        first = end;
        words = 0;
    };
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t item_words = words_of(i);
        if (words > 0 && words + item_words > budget_words) close(i);
        words += item_words;
    }
    close(count);
    return split;
}
