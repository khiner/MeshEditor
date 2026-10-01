#pragma once
#include "gpu/Types.h"

GPU_CONSTANT uint32_t PoseAttributeRadixBits = 12u;
GPU_CONSTANT uint32_t PoseAttributeRadixMask = (1u << PoseAttributeRadixBits) - 1u;

// Two radix levels address the 24-bit block part of a canonical record index.
// Interior entries name nodes.
// Leaves name 256-value payload blocks.
// Zero is absent.
// Each mapping node occupies one physical backing page.
struct PoseAttributeNode {
    GpuArray<uint32_t, 1u << PoseAttributeRadixBits> Children DEFAULT();
};
static_assert(sizeof(PoseAttributeNode) == 16384);
