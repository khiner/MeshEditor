#pragma once

#include "gpu/Types.h"
#include <cassert>
#include <functional>
#include <span>
#include <vector>

struct Range {
    uint32_t Offset{0}, Count{0};
};

constexpr uint32_t OffsetOrInvalid(Range range) { return range.Count > 0 ? range.Offset : InvalidOffset; }

inline void AppendRange(std::vector<Range> &ranges, Range range) {
    if (!range.Count) return;
    if (!ranges.empty() && ranges.back().Offset + ranges.back().Count == range.Offset) ranges.back().Count += range.Count;
    else ranges.push_back(range);
}
void CoalesceRanges(std::vector<Range> &);

// Copy surviving runs to their compacted positions after removing sorted indices.
template<typename T, typename Index = std::identity>
inline void ForEachSurvivorRun(Range active, const T &indices, auto &&copy, Index index_of = {}) {
    auto read = active.Offset, write = active.Offset;
    for (const auto &entry : indices) {
        const auto index = std::invoke(index_of, entry);
        assert(index >= read && index < active.Offset + active.Count);
        copy(read, write, index - read);
        write += index - read;
        read = index + 1u;
    }
    copy(read, write, active.Offset + active.Count - read);
}

// Visits maximal consecutive runs as (first entry, entry count), retaining
// entry positions for callers with parallel payload arrays.
template<typename T, typename Index = std::identity>
inline void ForEachIndexRun(const T &indices, auto &&visit, Index index = {}) {
    for (size_t first = 0; first < indices.size();) {
        auto end = first + 1;
        while (end < indices.size() && uint64_t(std::invoke(index, indices[end])) == uint64_t(std::invoke(index, indices[first])) + end - first) ++end;
        visit(first, end - first);
        first = end;
    }
}
