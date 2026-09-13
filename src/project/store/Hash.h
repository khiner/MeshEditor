#pragma once

#include <cstdint>
#include <cstring>
#include <span>

// Non-cryptographic hashes for corruption detection and content deduplication.
namespace store {
struct Hash128 {
    uint64_t A{}, B{};
    bool operator==(const Hash128 &) const = default;
};

// H sums the Term values for non-default slots.
struct Stamp {
    Hash128 H{};
    uint64_t Length{};
    bool operator==(const Stamp &) const = default;
};

inline uint64_t Mix64(uint64_t x) {
    x ^= x >> 32;
    x *= 0xe9846af9b1a615dull;
    x ^= x >> 32;
    x *= 0xe9846af9b1a615dull;
    x ^= x >> 28;
    return x;
}

inline uint64_t MulFold(uint64_t a, uint64_t b) {
    const auto p = static_cast<unsigned __int128>(a) * b;
    return uint64_t(p) ^ uint64_t(p >> 64);
}

// Identical bytes produce the same hash at every slot.
inline Hash128 HashBytes(std::span<const std::byte> bytes) {
    constexpr uint64_t C0 = 0x9e3779b97f4a7c15ull, C1 = 0xbf58476d1ce4e5b9ull, C2 = 0x94d049bb133111ebull, C3 = 0x2545f4914f6cdd1dull;
    uint64_t a = C0 ^ (bytes.size() * C1), b = C2 ^ (bytes.size() * C3);
    const auto *p = bytes.data();
    auto n = bytes.size();
    uint64_t w0, w1;
    while (n >= 16) {
        std::memcpy(&w0, p, 8);
        std::memcpy(&w1, p + 8, 8);
        a = MulFold(w0 ^ a, C1);
        b = MulFold(w1 ^ b, C3);
        p += 16;
        n -= 16;
    }
    w0 = w1 = 0;
    if (n >= 8) {
        std::memcpy(&w0, p, 8);
        p += 8;
        n -= 8;
    }
    if (n) std::memcpy(&w1, p, n);
    a = MulFold(w0 ^ a, C2);
    b = MulFold(w1 ^ b, C0);
    return {Mix64(a + b), Mix64(a ^ Mix64(b))};
}

// Slot-dependent terms for two 64-bit state-hash sums.
struct Term {
    uint64_t L0, L1;
    Term(uint64_t slot, Hash128 h)
        : L0(Mix64(h.A ^ Mix64(slot + 0x9e3779b97f4a7c15ull))), L1(Mix64(h.B ^ Mix64(slot + 0xc2b2ae3d27d4eb4full))) {}
};

struct Hash128Hasher {
    size_t operator()(const Hash128 &h) const { return size_t(h.A ^ h.B); }
};
} // namespace store
