#pragma once
#include <algorithm>
#include <cstdint>

// The lookup key of the edge between two vertices, with the lower handle in the high word.
inline uint64_t EdgeKey(uint32_t a, uint32_t b) { return uint64_t(std::min(a, b)) << 32u | std::max(a, b); }
