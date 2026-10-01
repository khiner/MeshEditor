#pragma once

#include "gpu/SlotOffset.h"
#include "gpu/Types.h"

// Sparse blocks contain an address block key plus eight 32-bit masks. The
// active-slot list dispatches one 256-lane tile per occupied block. Inclusive
// compact prefixes are keyed by hash slot, supporting rank and select.
GPU_CONSTANT uint32_t WorkHeaderWords = 8u;
GPU_CONSTANT uint32_t WorkBlockWords = 9u;
inline uint32_t WorkHash(uint32_t block, uint32_t capacity) {
    block ^= block >> 16u;
    block *= 0x7feb352du;
    block ^= block >> 15u;
    block *= 0x846ca68bu;
    block ^= block >> 16u;
    return block & (capacity - 1u);
}

struct ElementWork {
    SlotOffset Storage DEFAULT();
    uint32_t Count DEFAULT(); // Address-domain bound, independent of allocation size
    uint32_t Capacity DEFAULT(); // Power-of-two hash slots, sized by affected blocks
};
static_assert(sizeof(ElementWork) == 16, "ElementWork size");
