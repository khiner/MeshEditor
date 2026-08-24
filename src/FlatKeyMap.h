#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <vector>

// Open-addressed map from a fixed-width key to a uint32. Reset fixes the key width in words.
// A caller that consumes each entry once marks the value Taken, and later probes pass that slot by.
struct FlatKeyMap {
    static constexpr uint32_t Taken{std::numeric_limits<uint32_t>::max()};

    void Reset(uint32_t key_words, uint32_t expected_keys) {
        KeyWords = key_words;
        Keys.clear();
        Values.clear();
        Keys.reserve(size_t(expected_keys) * key_words);
        Values.reserve(expected_keys);
        Table.assign(std::bit_ceil(std::max<size_t>(size_t(expected_keys) * 2u, 64u)), 0u);
        Mask = Table.size() - 1u;
    }

    uint32_t *Find(const uint32_t *key) {
        for (auto i = Hash(key) & Mask; Table[i] != 0u; i = (i + 1u) & Mask) {
            if (const uint32_t slot = Table[i] - 1u; Values[slot] != Taken && Equal(slot, key)) return &Values[slot];
        }
        return nullptr;
    }

    void Insert(const uint32_t *key, uint32_t value) {
        if ((Values.size() + 1u) * 2u > Table.size()) Grow();
        Keys.insert(Keys.end(), key, key + KeyWords);
        Values.push_back(value);
        Place(uint32_t(Values.size() - 1u));
    }

private:
    void Place(uint32_t slot) {
        auto i = Hash(&Keys[size_t(slot) * KeyWords]) & Mask;
        while (Table[i] != 0u) i = (i + 1u) & Mask;
        Table[i] = slot + 1u;
    }
    void Grow() {
        Table.assign(Table.size() * 2u, 0u);
        Mask = Table.size() - 1u;
        for (uint32_t slot = 0; slot < Values.size(); ++slot) Place(slot);
    }
    bool Equal(uint32_t slot, const uint32_t *key) const {
        return std::equal(key, key + KeyWords, Keys.begin() + size_t(slot) * KeyWords);
    }
    size_t Hash(const uint32_t *key) const {
        uint64_t h = 0xcbf29ce484222325ull;
        for (uint32_t w = 0; w < KeyWords; ++w) h = (h ^ key[w]) * 0x100000001b3ull;
        // Finalize, so the low bits the table indexes with depend on every word.
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdull;
        h ^= h >> 33;
        return size_t(h);
    }

    std::vector<uint32_t> Table, Keys, Values;
    size_t Mask{};
    uint32_t KeyWords{};
};
