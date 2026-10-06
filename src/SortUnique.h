#pragma once

#include <algorithm>

// Sorts `values` ascending and drops repeats.
void SortUnique(auto &values) {
    std::ranges::sort(values);
    values.erase(std::ranges::unique(values).begin(), values.end());
}
