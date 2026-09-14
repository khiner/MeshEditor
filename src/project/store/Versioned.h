#pragma once

#include "project/store/LiveTrie.h"

#include <cassert>
#include <cstring>
#include <span>
#include <vector>

namespace store {
// Versioned CPU bytes with growing capacity and zero-filled storage beyond Len.
struct VersionedBuffer;
inline Live BufferSlots(VersionedBuffer &);

struct VersionedBuffer {
    explicit VersionedBuffer(uint32_t page_bytes = 512, uint32_t levels = 5)
        : PageBytes(page_bytes), Trie(BufferSlots(*this), levels, page_bytes) {}

    uint64_t Size() const { return Len; }
    const std::byte *Data() const { return Bytes.data(); }

    // Capture old bytes and return a writable span.
    std::span<std::byte> Mutable(uint64_t offset, uint64_t size) {
        Write(offset, size);
        return {Bytes.data() + offset, size};
    }
    // Capture old bytes and grow capacity to cover the range.
    void Write(uint64_t offset, uint64_t size) {
        if (size == 0) return;
        Reserve(offset + size);
        const auto first = offset / PageBytes, last = (offset + size - 1) / PageBytes;
        Trie.Write(first, last - first + 1);
    }
    // Capture removed bytes and zero storage beyond the new length.
    void Resize(uint64_t len) {
        if (len > Len) {
            Write(Len, len - Len);
        } else if (len < Len) {
            Write(len, Len - len);
            std::memset(Bytes.data() + len, 0, Len - len);
        }
        Len = len;
    }
    void Reserve(uint64_t bytes) {
        const auto pages = SlotsFor(Trie.L, bytes);
        if (pages * PageBytes > Bytes.size()) Bytes.resize(pages * PageBytes);
    }

    uint32_t PageBytes;
    std::vector<std::byte> Bytes;
    uint64_t Len{};
    LiveTrie Trie;
};

inline Live BufferSlots(VersionedBuffer &self) {
    Live live;
    live.PageBytes = self.PageBytes;
    live.Length = [&self] { return self.Len; };
    live.SetLength = [&self](uint64_t len) {
        self.Reserve(len);
        self.Len = len;
    };
    live.Present = [&self](uint64_t page) { return (page + 1) * self.PageBytes <= self.Bytes.size(); };
    live.Read = [&self](uint64_t page) -> std::span<const std::byte> { return {self.Bytes.data() + page * self.PageBytes, self.PageBytes}; };
    live.Replace = [&self](uint64_t page, Blob incoming, bool &was_present) {
        self.Reserve((page + 1) * self.PageBytes);
        was_present = true;
        return SwapBlob({self.Bytes.data() + page * self.PageBytes, self.PageBytes}, incoming);
    };
    live.Erase = [&self](uint64_t page) {
        const auto old = Capture(self.Trie.L, page);
        std::memset(self.Bytes.data() + page * self.PageBytes, 0, self.PageBytes);
        return old;
    };
    live.ForEachPresent = [&self](const std::function<void(uint64_t)> &fn) {
        for (uint64_t p = 0, n = SlotsFor(self.Trie.L, self.Len); p < n; ++p) fn(p);
    };
    return live;
}

template<typename T>
struct VersionedVector {
    static_assert(std::is_trivially_copyable_v<T>);

    explicit VersionedVector(uint32_t page_bytes = 512) : Buffer(page_bytes) {}

    size_t size() const { return Buffer.Size() / sizeof(T); }
    bool empty() const { return size() == 0; }
    std::span<const T> View() const { return {reinterpret_cast<const T *>(Buffer.Data()), size()}; }
    const T &operator[](size_t i) const { return View()[i]; }

    T &Mutable(size_t i) {
        assert(i < size());
        return *reinterpret_cast<T *>(Buffer.Mutable(i * sizeof(T), sizeof(T)).data());
    }
    void Set(size_t i, const T &v) { Mutable(i) = v; }
    void PushBack(const T &v) {
        const auto i = size();
        Buffer.Resize((i + 1) * sizeof(T));
        std::memcpy(Buffer.Bytes.data() + i * sizeof(T), &v, sizeof(T));
    }
    void PopBack() { Buffer.Resize((size() - 1) * sizeof(T)); }
    T Back() const { return (*this)[size() - 1]; }
    void Clear() { Buffer.Resize(0); }

    VersionedBuffer Buffer;
};
} // namespace store
