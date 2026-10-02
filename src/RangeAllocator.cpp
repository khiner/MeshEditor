#include "RangeAllocator.h"

void CoalesceRanges(std::vector<Range> &ranges) {
    std::erase_if(ranges, [](Range range) { return range.Count == 0u; });
    if (!std::ranges::is_sorted(ranges, {}, &Range::Offset)) std::ranges::sort(ranges, {}, &Range::Offset);
    size_t count = 0u;
    for (const auto range : ranges) {
        if (count && ranges[count - 1u].Offset + ranges[count - 1u].Count == range.Offset) ranges[count - 1u].Count += range.Count;
        else ranges[count++] = range;
    }
    ranges.resize(count);
}

void RangeAllocator::Free(std::vector<Range> ranges) {
    CoalesceRanges(ranges);
    for (const auto range : ranges) Free(range);
}
