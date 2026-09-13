#pragma once

#include "project/store/History.h"

#include <algorithm>
#include <optional>
#include <zpp_bits.h>

namespace project {
// Track an externally owned vector and resize it after restoring its slots.
template<typename T> struct VectorHistory;
template<typename T> store::Live VectorSlots(VectorHistory<T> &);

template<typename T> struct VectorHistory {
    std::vector<T> &Values;
    std::optional<size_t> OriginalLength;
    std::vector<std::byte> Scratch;
    store::LiveTrie Trie;

    VectorHistory(std::vector<T> &values, store::History &history, std::string name, int phase = 0)
        : Values(values), Trie(VectorSlots(*this), 4, std::is_trivially_copyable_v<T> ? sizeof(T) : 0) {
        history.Track(Trie, std::move(name), phase);
        Trie.Write(0, values.size());
    }
};

template<typename T> store::Live VectorSlots(VectorHistory<T> &self) {
    store::Live live;
    live.Length = [&self] { return self.OriginalLength.value_or(self.Values.size()); };
    live.SetLength = [&self](uint64_t size) {
        self.Values.resize(size);
        self.OriginalLength.reset();
    };
    live.Present = [&self](uint64_t i) { return i < self.OriginalLength.value_or(self.Values.size()); };
    live.Read = [&self](uint64_t i) -> std::span<const std::byte> {
        if constexpr (std::is_trivially_copyable_v<T>) return std::as_bytes(std::span{&self.Values[i], 1});
        else {
            self.Scratch.clear();
            zpp::bits::out out{self.Scratch};
            // zpp reflects large aggregates through non-const references.
            out(self.Values[i]).or_throw();
            self.Scratch.resize(out.position());
            return self.Scratch;
        }
    };
    live.Replace = [&self](uint64_t i, store::Blob incoming, bool &was_present) {
        if (!self.OriginalLength) self.OriginalLength = self.Values.size();
        was_present = self.Trie.L.Present(i);
        const auto old = was_present ? store::Capture(self.Trie.L, i) : store::Blob{};
        if (self.Values.size() <= i) self.Values.resize(i + 1);
        if constexpr (std::is_trivially_copyable_v<T>) std::memcpy(&self.Values[i], incoming.Data, sizeof(T));
        else zpp::bits::in{incoming.View()}(self.Values[i]).or_throw();
        store::FreeBlob(incoming);
        return old;
    };
    live.Erase = [&self](uint64_t i) {
        const auto old = store::Capture(self.Trie.L, i);
        self.Values[i] = {};
        return old;
    };
    return live;
}

} // namespace project
