#pragma once

#include "gpu/Types.h"

// The lowbias32 integer mixer, whose output bits each depend on every input bit.
inline uint32_t HashMix(uint32_t value) {
    value = (value ^ (value >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}
inline uint32_t HashPair(uint32_t a, uint32_t b) { return HashMix(a * 0x9e3779b1u ^ b * 0x85ebca77u); }

GPU_CONSTANT uint32_t KeyTableWords = 3u; // An open-addressed key table slot: the key's two words, then its value
