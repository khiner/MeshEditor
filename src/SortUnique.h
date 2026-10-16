#pragma once

#include <algorithm>

void SortUnique(auto &values) {
    std::ranges::sort(values);
    values.erase(std::ranges::unique(values).begin(), values.end());
}
