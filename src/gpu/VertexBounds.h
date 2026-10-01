#pragma once
#include "gpu/Types.h"

// A leaf covers 256 canonical vertices. Three 256-way parent levels cover
// the entire 32-bit vertex address space without mesh-relative ranks.
GPU_CONSTANT uint32_t VertexBoundsLevels = 4u;
// Bounds are queried per vertex block. Compact mapping nodes and 32-value
// payload blocks avoid the hot pose streams' larger tables for small owners.
struct VertexBoundsMapNode {
    GpuArray<uint32_t,128> Children DEFAULT();
};
static_assert(sizeof(VertexBoundsMapNode) == 512);
inline uint32_t VertexBoundsKey(uint32_t level, uint32_t index) {
    return index + (level == 0u ? 0u : level == 1u ? 0x1000000u : level == 2u ? 0x1010000u : 0x1010100u);
}
