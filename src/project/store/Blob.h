#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace store {
// Release malloc-allocated data with FreeBlob.
struct Blob {
    std::byte *Data{};
    uint32_t Size{};

    std::span<const std::byte> View() const { return {Data, Size}; }
};

inline bool IsZero(std::span<const std::byte> bytes) {
    return bytes.empty() || (bytes.front() == std::byte{} && std::memcmp(bytes.data(), bytes.data() + 1, bytes.size() - 1) == 0);
}

inline Blob AllocBlob(uint32_t size) {
    return {size ? static_cast<std::byte *>(std::malloc(size)) : nullptr, size};
}
inline Blob CopyBlob(std::span<const std::byte> bytes) {
    auto b = AllocBlob(uint32_t(bytes.size()));
    if (b.Size) std::memcpy(b.Data, bytes.data(), b.Size);
    return b;
}
// Requires equal live and incoming sizes.
inline Blob SwapBlob(std::span<std::byte> live, Blob incoming, std::vector<std::byte> &scratch) {
    assert(live.size() == incoming.Size);
    scratch.resize(live.size());
    std::memcpy(scratch.data(), live.data(), live.size());
    std::memcpy(live.data(), incoming.Data, live.size());
    std::memcpy(incoming.Data, scratch.data(), live.size());
    return incoming;
}
inline void FreeBlob(Blob &b) {
    std::free(b.Data);
    b = {};
}
} // namespace store
