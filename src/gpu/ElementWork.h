#pragma once

#include "gpu/Hash.h"
#include "gpu/SlotOffset.h"

// Sparse blocks contain an address block key plus eight 32-bit masks.
// The active-slot list dispatches one 256-lane tile per occupied block.
// Inclusive compact prefixes are keyed by hash slot, supporting rank and select.
GPU_CONSTANT uint32_t WorkHeaderWords = 8u;
GPU_CONSTANT uint32_t WorkBlockWords = 9u;
inline uint32_t WorkHash(uint32_t block, uint32_t capacity) { return HashMix(block) & (capacity - 1u); }

struct ElementWork {
    SlotOffset Storage DEFAULT();
    uint32_t Count DEFAULT(); // Address-domain bound, independent of allocation size
    uint32_t Capacity DEFAULT(); // Power-of-two hash slots, sized by affected blocks
};

// A record's ascending blocks in one element domain, followed by their inclusive live prefix, and the domain's block membership.
// A dispatch over them runs one thread per slot of the Count blocks.
struct DomainBlocks {
    SlotOffset Blocks DEFAULT();
    uint32_t Count DEFAULT(), MembershipSlot DEFAULT(InvalidSlot);
};
