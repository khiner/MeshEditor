#pragma once

#include "gpu/Element.h"

#include <span>

struct Mesh;

namespace selection {

// Every function here takes one mesh's own selection-bit words, where element `handle` is bit
// `handle % 32` of word `handle / 32`, and `count`, the element count of the current edit mode.
// Each store range is sized to its own element domain, so `count` never exceeds it.

void ForEachSelected(std::span<const uint32_t> bits, uint32_t count, auto &&fn) {
    const uint32_t last_word = (count + 31) / 32;
    for (uint32_t w = 0; w < last_word; ++w) {
        uint32_t word = bits[w];
        while (word) {
            const uint32_t handle = w * 32 + __builtin_ctz(word);
            if (handle < count) fn(handle);
            word &= word - 1;
        }
    }
}
uint32_t GetElementCount(const Mesh &, Element);

} // namespace selection
