#pragma once

#include "Range.h"
#include "metal/Bindless.h"

#include <deque>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace store {
struct History;
struct Pages;
} // namespace store

template<typename T>
constexpr std::span<const std::byte> as_bytes(const std::vector<T> &v) { return std::as_bytes(std::span{v}); }
template<typename T, uint32_t N>
constexpr std::span<const std::byte> as_bytes(const std::array<T, N> &v) { return std::as_bytes(std::span{v}); }
template<typename T>
constexpr std::span<const std::byte> as_bytes(const T &v) { return {reinterpret_cast<const std::byte *>(&v), sizeof(T)}; }

namespace mtl {
struct SparseBuffer;
// Canonical storage has stable CPU views and versioned physical pages.
// Workspaces have no history or page clones, and their growth preserves the used bytes and zeroes the rest.
// Growth invalidates borrowed CPU spans and direct GPU bindings, so finish or reserve them before encoding consumers.
enum class BufferLifetime { Canonical, Workspace };
// Tracked buffers capture and restore history in pages of this size.
inline constexpr uint64_t HistoryPageBytes = 16u << 10;
// Commit GPU consumers before retiring their owners. Slots, virtual mappings,
// and physical pages survive together until the retirement fence completes.
struct BufferContext {
    BufferContext(const Context &, BindlessSet &);
    ~BufferContext();
    BufferContext(const BufferContext &) = delete;
    BufferContext(BufferContext &&) = delete;

    // Fences the buffers retired since the last reclaim behind committed GPU work and releases the batches whose fence completed.
    // Chain submits, finished frames and scene resets reclaim, so allocations never wait for retired readers.
    bool ReclaimRetiredBuffers(bool wait = false);
    // A free slot, reclaiming retired readers' slots only when the table is full.
    uint32_t AllocateSlot(SlotType);
    // A workspace of at least `bytes` that starts with `prefix` and reads zero after it.
    NS::SharedPtr<MTL::Buffer> AcquireWorkspace(uint64_t bytes, std::span<const std::byte> prefix);

    std::string DebugHeapUsage() const;

    const Context &Ctx;
    BindlessSet &Slots;
    struct RetiredBuffer {
        std::shared_ptr<SparseBuffer> Storage;
        TypedSlot Binding{SlotType::Buffer, InvalidSlot};
        NS::SharedPtr<MTL::Buffer> Workspace;
    };
    std::vector<RetiredBuffer> Retired;
    struct RetirementBatch {
        std::vector<RetiredBuffer> Buffers;
        NS::SharedPtr<MTL::CommandBuffer> Fence;
    };
    std::deque<RetirementBatch> Retirements;
private:
    void RecycleWorkspace(NS::SharedPtr<MTL::Buffer>);
    std::array<std::vector<NS::SharedPtr<MTL::Buffer>>,64> WorkspaceCache;
    uint64_t CachedWorkspaceBytes{};
};

// A zero size defers allocation.
NS::SharedPtr<MTL::Buffer> NewBuffer(const Context &, uint64_t size);

struct Buffer;
// Increasing unique resident history pages of one canonical buffer.
struct BufferFootprint {
    const Buffer *Source;
    std::span<const uint32_t> Pages;
};
// Copies each footprint's history pages on the CPU into a read-only clone of its source.
// A clone maps the physical pages holding its footprint at the source's offsets, reads its unmapped pages as zero, and has no CPU view.
// Only the footprint's bytes are copied, so readers read only those.
// Later source writes leave the clone unchanged.
// Submit a clone's readers before destroying it. Its slot and pages retire after those readers complete.
std::vector<Buffer> CloneFootprints(BufferContext &, std::span<const BufferFootprint>);

struct Buffer {
    Buffer(BufferContext &, uint64_t size, SlotType, BufferLifetime = BufferLifetime::Canonical);
    Buffer(BufferContext &, std::span<const std::byte>, SlotType, BufferLifetime = BufferLifetime::Canonical);
    Buffer(BufferContext &, uint64_t size);

    Buffer(const Buffer &) = delete;
    Buffer(Buffer &&) noexcept;
    Buffer &operator=(const Buffer &) = delete;
    Buffer &operator=(Buffer &&) noexcept;
    ~Buffer();

    void Update(std::span<const std::byte>, uint64_t offset = 0);
    void Reserve(uint64_t);
    void SetUsedSize(uint64_t);
    void CaptureWrite(uint64_t offset, uint64_t size) const;
    // Capture a complete sparse write footprint.
    // History pages are increasing and unique, and element indices may be unordered or repeated.
    void CaptureWritePages(std::span<const uint32_t> pages) const;
    void CaptureWriteElements(std::span<const uint32_t> elements, uint32_t stride) const;
    void CaptureWriteRanges(std::span<const Range> ranges, uint32_t stride) const;
    void Track(store::History &, std::string name);
    store::Pages *History() const { return Tracked.get(); }

    MTL::Buffer *operator*() const;
    std::span<std::byte> Contents() const;
    void Move(uint64_t from, uint64_t to, uint64_t size) const;
    std::span<std::byte> GetMutableRange(uint64_t offset, uint64_t size) const;
    template<typename T> std::span<T> SetCount(uint32_t count) {
        const auto size = uint64_t(count) * sizeof(T);
        SetUsedSize(size);
        if (count == 0) return {};
        return {reinterpret_cast<T *>(GetMutableRange(0, size).data()), count};
    }
    template<typename T> uint32_t Count() const { return uint32_t(UsedSize / sizeof(T)); }
    template<typename T> uint32_t Append(const T &value) {
        const auto index = Count<T>();
        Update(as_bytes(value), uint64_t(index) * sizeof(T));
        return index;
    }
    template<typename T> std::span<const T> GetSpan(Range range) const {
        if (range.Count == 0) return {};
        return {reinterpret_cast<const T *>(Contents().data()) + range.Offset, range.Count};
    }
    template<typename T> std::span<const T> GetSpan() const { return GetSpan<T>({0, Count<T>()}); }
    template<typename T> std::span<T> GetMutableSpan(Range range) const {
        if (range.Count == 0) return {};
        return {reinterpret_cast<T *>(GetMutableRange(uint64_t(range.Offset) * sizeof(T), uint64_t(range.Count) * sizeof(T)).data()), range.Count};
    }
    template<typename T> std::span<T> GetMutableSpan() const { return GetMutableSpan<T>({0, Count<T>()}); }

    BufferContext &Ctx;
    uint32_t Slot{InvalidSlot};
    uint64_t UsedSize{0};

private:
    friend std::vector<Buffer> CloneFootprints(BufferContext &, std::span<const BufferFootprint>);
    std::shared_ptr<SparseBuffer> Storage;
    NS::SharedPtr<MTL::Buffer> Workspace;
    std::vector<NS::SharedPtr<MTL::Buffer>> PreviousWorkspaces;
    BufferLifetime Lifetime{BufferLifetime::Canonical};
    std::unique_ptr<store::Pages> Tracked;
    void Retire();
    void UpdateSlot();

    SlotType Type{};
};

} // namespace mtl
