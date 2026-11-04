#pragma once

// A triangle or quad's per-side values, the fourth zero for a triangle.
#ifdef __METAL_VERSION__
#include "gpu/Types.h"
using PartitionSides = uint4;
#else
#include <array>
#include <cstdint>
using PartitionSides = std::array<uint32_t, 4>;
#endif

// Manifold's sort of a face's divisions: descending for a triangle, and for a quad rotated to its least side, breaking ties by the side after it.
// Order holds the input side at each sorted position.
struct SortedDivisions {
    PartitionSides Sorted, Order;
};
inline SortedDivisions SortDivisions(PartitionSides divisions) {
    SortedDivisions result{divisions, divisions};
    for (uint32_t i = 0u; i < 4u; ++i) result.Order[i] = i;
    if (divisions[3] == 0u) {
        const auto swap = [&](uint32_t i) {
            const uint32_t value = result.Sorted[i], side = result.Order[i];
            result.Sorted[i] = result.Sorted[i + 1u];
            result.Order[i] = result.Order[i + 1u];
            result.Sorted[i + 1u] = value;
            result.Order[i + 1u] = side;
        };
        if (result.Sorted[2] > result.Sorted[1]) swap(1u);
        if (result.Sorted[1] > result.Sorted[0]) {
            swap(0u);
            if (result.Sorted[2] > result.Sorted[1]) swap(1u);
        }
        return result;
    }
    uint32_t least = 0u;
    for (uint32_t i = 1u; i < 4u; ++i) {
        if (divisions[i] < divisions[least] || (divisions[i] == divisions[least] && divisions[(i + 1u) % 4u] < divisions[(least + 1u) % 4u])) least = i;
    }
    for (uint32_t i = 0u; i < 4u; ++i) {
        result.Order[i] = (i + least) % 4u;
        result.Sorted[i] = divisions[result.Order[i]];
    }
    return result;
}
