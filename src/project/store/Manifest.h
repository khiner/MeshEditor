#pragma once

#include "project/store/Hash.h"
#include <array>

namespace store {
constexpr uint32_t Bits = 6, Fanout = 1u << Bits;
constexpr uint64_t SlotSpan(uint32_t level) { return uint64_t{1} << (Bits * level); }

// Zero hashes represent absent or default subtrees.
using ManifestChildren = std::array<Hash128, Fanout>;
struct ManifestRecord {
    std::array<std::byte, 3 + Fanout * (1 + sizeof(Hash128))> Bytes;
    uint32_t Size{3};
    ManifestRecord(uint32_t level, const ManifestChildren &children) {
        Bytes[0] = std::byte(level);
        uint16_t count = 0;
        for (uint32_t i = 0; i < Fanout; ++i) {
            if (children[i] == Hash128{}) continue;
            Bytes[Size++] = std::byte(i);
            std::memcpy(Bytes.data() + Size, &children[i], sizeof(Hash128));
            Size += sizeof(Hash128);
            ++count;
        }
        std::memcpy(Bytes.data() + 1, &count, sizeof(count));
    }
    std::span<const std::byte> View() const { return {Bytes.data(), Size}; }
    Hash128 Hash() const { return Size == 3 ? Hash128{} : HashBytes(View()); }
};
} // namespace store
