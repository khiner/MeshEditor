#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace store {
// Byte or native value ownership; release with FreeBlob.
struct Blob {
    std::byte *Data{};
    uint32_t Size{};
    void (*Destroy)(void *){};
    uint64_t NativeBytes{};

    uint64_t OwnedBytes() const { return Destroy ? NativeBytes : Size; }

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
inline Blob SwapBlob(std::span<std::byte> live, Blob incoming) {
    assert(live.size() == incoming.Size);
    std::swap_ranges(live.begin(), live.end(), incoming.Data);
    return incoming;
}
inline void FreeBlob(Blob &b) {
    if (b.Destroy) b.Destroy(b.Data);
    else std::free(b.Data);
    b = {};
}
} // namespace store
