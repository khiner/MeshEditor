#pragma once

#include "gpu/Types.h"
#include <span>

struct Range {
    uint32_t Offset{0}, Count{0};
};

constexpr uint32_t OffsetOrInvalid(Range range) { return range.Count > 0 ? range.Offset : InvalidOffset; }

// Visits maximal consecutive runs as (first entry, entry count), retaining
// entry positions for callers with parallel payload arrays.
inline void ForEachIndexRun(std::span<const uint32_t> indices, auto &&visit) {
    for (size_t first = 0; first < indices.size();) {
        auto end = first + 1;
        while (end < indices.size() && uint64_t(indices[end]) == uint64_t(indices[first]) + end - first) ++end;
        visit(first, end - first);
        first = end;
    }
}
