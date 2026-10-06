#include "metal/Buffer.h"
#include "Parallel.h"
#include "Profile.h"
#include "metal/AutoreleaseScope.h"

#include "metal/MetalCpp.h"
#include "metal/SparseBuffer.h"
#include "project/store/History.h"
#include "project/store/Pages.h"

#include <format>

namespace mtl {
namespace {
// Copies `source`, or clears when it is null, in parallel chunks.
void CopyBytes(std::byte *destination, const std::byte *source, uint64_t bytes) {
    constexpr uint64_t Chunk = 4u << 20;
    ParallelFor(uint32_t((bytes + Chunk - 1u) / Chunk), [&](uint32_t i) {
        const auto first = uint64_t(i) * Chunk, count = std::min(Chunk, bytes - first);
        if (source) std::memcpy(destination + first, source + first, count);
        else std::memset(destination + first, 0, count);
    });
}
} // namespace

BufferContext::BufferContext(const Context &ctx, BindlessSet &slots) : Ctx(ctx), Slots(slots) {}
BufferContext::~BufferContext() {
    const AutoreleaseScope pool;
    ReclaimRetiredBuffers(true);
    for (const auto &bin : WorkspaceCache)
        for (const auto &buffer : bin) Ctx.RemoveResident(buffer.get());
    AutoreleaseScope::Release(WorkspaceCache, Retirements, Retired);
}

NS::SharedPtr<MTL::Buffer> BufferContext::AcquireWorkspace(uint64_t bytes, std::span<const std::byte> prefix) {
    if (bytes > std::bit_floor(uint64_t(Ctx.Device->maxBufferLength()))) throw std::length_error("Metal workspace address space exhausted.");
    const auto capacity = std::bit_ceil(std::max(bytes, uint64_t{16u << 10}));
    if (!capacity || capacity > Ctx.Device->maxBufferLength()) throw std::length_error("Metal workspace address space exhausted.");
    auto &bin = WorkspaceCache[std::countr_zero(capacity)];
    const bool recycled = !bin.empty();
    NS::SharedPtr<MTL::Buffer> result;
    if (recycled) {
        result = std::move(bin.back());
        bin.pop_back();
        CachedWorkspaceBytes -= capacity;
    } else {
        // A new Metal buffer starts zeroed.
        result = NewBuffer(Ctx, capacity);
        Ctx.AddResident(result.get());
    }
    auto *contents = static_cast<std::byte *>(result->contents());
    CopyBytes(contents, prefix.data(), prefix.size());
    // A recycled workspace has no pending GPU users, so the host clears the bytes after the prefix.
    if (recycled) CopyBytes(contents + prefix.size(), nullptr, capacity - prefix.size());
    return result;
}

void BufferContext::RecycleWorkspace(NS::SharedPtr<MTL::Buffer> buffer) {
    constexpr uint64_t cache_limit = 32u << 20;
    const auto bytes = uint64_t(buffer->length());
    if (bytes > cache_limit - CachedWorkspaceBytes) {
        Ctx.RemoveResident(buffer.get());
        return;
    }
    WorkspaceCache[std::countr_zero(bytes)].push_back(std::move(buffer));
    CachedWorkspaceBytes += bytes;
}
void BufferContext::Release(std::span<RetiredBuffer> buffers) {
    for (auto &buffer : buffers) {
        if (buffer.Binding.Slot != InvalidSlot) Slots.Release(buffer.Binding);
        if (buffer.Workspace) RecycleWorkspace(std::move(buffer.Workspace));
    }
}

bool BufferContext::ReclaimRetiredBuffers(bool wait) {
    const AutoreleaseScope pool;
    bool released = false;
    bool completed = true;
    if (!Retired.empty()) {
        // Mapping work can still be pending after the last render submit.
        // Publish it before fencing the storage and slots that it references.
        auto *fence = Ctx.Queue->commandBuffer();
        Ctx.OrderAfterGpuWork(fence);
        FenceRetiredBuffers(fence);
        Commit(fence);
    }
    while (!Retirements.empty()) {
        auto &batch = Retirements.front();
        if (wait) batch.Fence->waitUntilCompleted();
        const auto status = batch.Fence->status();
        if (status != MTL::CommandBufferStatusCompleted && status != MTL::CommandBufferStatusError) break;
        completed &= status == MTL::CommandBufferStatusCompleted;
        Release(batch.Buffers);
        Retirements.pop_front();
        released = true;
    }
    if (released) Ctx.CommitResidency();
    if (wait) completed &= Ctx.DrainMappings();
    return completed;
}

void BufferContext::FenceRetiredBuffers(MTL::CommandBuffer *command) {
    if (!Retired.empty()) Retirements.push_back({std::exchange(Retired, {}), NS::RetainPtr(command)});
}

void BufferContext::ReleaseRetiredBuffers(size_t first) {
    if (Retired.size() <= first) return;
    const AutoreleaseScope pool;
    // The next residency commit drops recycled workspaces that leave the cache.
    Release(std::span{Retired}.subspan(first));
    Retired.resize(first);
}

uint32_t BufferContext::AllocateSlot(SlotType type) {
    const auto slot = Slots.TryAllocate(type);
    if (slot != InvalidSlot) return slot;
    // Retired readers hold their slots until a reclaim, which a full table brings forward.
    if (!Retirements.empty() || !Retired.empty()) ReclaimRetiredBuffers(true);
    return Slots.Allocate(type);
}

NS::SharedPtr<MTL::Buffer> NewBuffer(const Context &ctx, uint64_t size) {
    const AutoreleaseScope pool;
    if (size == 0) return {};
    auto buffer = NS::TransferPtr(ctx.Device->newBuffer(size, MTL::ResourceStorageModeShared));
    if (!buffer) throw std::runtime_error("Failed to allocate a Metal buffer.");
    return buffer;
}

std::string BufferContext::DebugHeapUsage() const {
    const AutoreleaseScope pool;
    static constexpr std::array<std::string_view, 6> Suffixes{"B", "KB", "MB", "GB", "TB", "PB"};
    const auto format_bytes = [](uint64_t bytes) {
        auto value = float(bytes);
        size_t pow = 0;
        for (; value >= 1024.f && pow + 1 < Suffixes.size(); ++pow) value /= 1024.f;
        return std::format("{:.2f} {}", value, Suffixes[pow]);
    };
    return std::format(
        "Device allocation:\n\tCurrent: {}\n\tRecommended maximum: {}\n",
        format_bytes(Ctx.Device->currentAllocatedSize()), format_bytes(Ctx.Device->recommendedMaxWorkingSetSize())
    );
}

Buffer::Buffer(BufferContext &ctx, uint64_t size, SlotType slot_type, BufferLifetime lifetime)
    : Ctx(ctx), Slot(ctx.AllocateSlot(slot_type)), Lifetime(lifetime), Type(slot_type) {
    Reserve(size);
}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, SlotType slot_type, BufferLifetime lifetime)
    : Buffer(ctx, data.size(), slot_type, lifetime) { Update(data); }

Buffer::Buffer(BufferContext &ctx, uint64_t size) : Ctx(ctx) { Reserve(size); }

Buffer::Buffer(Buffer &&other) noexcept
    : Ctx(other.Ctx), Slot(other.Slot), UsedSize(other.UsedSize),
      Storage(std::move(other.Storage)), Workspace(std::move(other.Workspace)), PreviousWorkspaces(std::move(other.PreviousWorkspaces)), Lifetime(other.Lifetime), Tracked(std::move(other.Tracked)), Type(other.Type) {
    if (Tracked) {
        Tracked->Backing = this;
        Tracked->Len = &UsedSize;
    }
    other.Slot = InvalidSlot;
}

Buffer &Buffer::operator=(Buffer &&other) noexcept {
    if (this != &other) {
        Retire();
        if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
        Slot = other.Slot;
        UsedSize = other.UsedSize;
        Storage = std::move(other.Storage);
        Workspace = std::move(other.Workspace);
        PreviousWorkspaces = std::move(other.PreviousWorkspaces);
        Lifetime = other.Lifetime;
        Type = other.Type;
        Tracked = std::move(other.Tracked);
        if (Tracked) {
            Tracked->Backing = this;
            Tracked->Len = &UsedSize;
        }
        other.Slot = InvalidSlot;
    }
    return *this;
}

Buffer::~Buffer() {
    Retire();
    if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
}

void Buffer::RetirePreviousWorkspaces() {
    for (auto &previous : PreviousWorkspaces) Ctx.Retired.push_back({.Workspace = std::move(previous)});
    PreviousWorkspaces.clear();
}

void Buffer::Retire() {
    RetirePreviousWorkspaces();
    if (Workspace) {
        Ctx.Retired.push_back({.Binding = {Type, Slot}, .Workspace = std::move(Workspace)});
        Slot = InvalidSlot;
    }
    if (Storage) {
        Ctx.Retired.push_back({std::move(Storage), {Type, Slot}});
        Slot = InvalidSlot;
    }
}

void Buffer::UpdateSlot() {
    if (Slot == InvalidSlot) return;
    Ctx.Slots.SetBuffer({Type, Slot}, **this);
}

MTL::Buffer *Buffer::operator*() const { return Workspace ? Workspace.get() : Storage ? Storage->Gpu.get() :
                                                                                        nullptr; }

std::span<std::byte> Buffer::Contents() const {
    return Workspace ? std::span{static_cast<std::byte *>(Workspace->contents()), Workspace->length()} : Storage ? Storage->Contents() :
                                                                                                                   std::span<std::byte>{};
}

void Buffer::Move(uint64_t from, uint64_t to, uint64_t size) const {
    if (!Storage && !Workspace) return;
    const auto allocated = Contents().size();
    if (size == 0 || from + size > allocated || to + size > allocated) return;
    CaptureWrite(to, size);
    auto *mapped = Contents().data();
    std::memmove(mapped + to, mapped + from, size);
}

std::span<std::byte> Buffer::GetMutableRange(uint64_t offset, uint64_t size) const {
    if (!Storage && !Workspace) return {};
    CaptureWrite(offset, size);
    return Contents().subspan(offset, size);
}

void Buffer::Reserve(uint64_t required_size) {
    if (required_size <= Contents().size()) return;
    if (Lifetime == BufferLifetime::Workspace) {
        const profile::CpuScope scope{"WorkspaceGrow"};
        const AutoreleaseScope pool;
        auto next = Ctx.AcquireWorkspace(required_size, Contents().first(UsedSize));
        // An encoder may retain a direct binding made before this growth.
        // Keep old allocations owned until the workspace's consumers retire.
        if (Workspace) PreviousWorkspaces.push_back(std::move(Workspace));
        Workspace = std::move(next);
        UpdateSlot();
        return;
    }
    const auto *previous = **this;
    if (!Storage) Storage = std::make_shared<SparseBuffer>(Ctx.Ctx, required_size);
    try {
        Storage->Reserve(required_size);
    } catch (...) {
        // Growth can replace the GPU address before a later page allocation
        // fails. Keep the bindless entry valid for the original resident data.
        if (previous != **this) UpdateSlot();
        throw;
    }
    if (previous != **this) UpdateSlot();
    if (Tracked) Tracked->Storage = Contents();
}

void Buffer::Update(std::span<const std::byte> data, uint64_t offset) {
    if (data.empty()) return;
    const auto required_size = offset + data.size();
    CaptureWrite(offset, data.size());
    SetUsedSize(std::max(UsedSize, required_size));
    std::memcpy(Contents().data() + offset, data.data(), data.size());
}

void Buffer::CaptureWrite(uint64_t offset, uint64_t size) const {
    if (!Tracked) return;
    const profile::CpuScope scope{"HistoryCapturePages"};
    Tracked->Write(offset, size);
}

void Buffer::CaptureWritePages(std::span<const uint32_t> pages) const {
    if (pages.empty()) return;
    for (size_t i = 0u; i < pages.size(); ++i)
        if (uint64_t(pages[i]) >= Contents().size() / HistoryPageBytes || (i && pages[i] <= pages[i - 1u])) {
            throw std::out_of_range("Write pages must be ordered, unique, and resident.");
        }
    if (!Tracked) return;
    const profile::CpuScope scope{"HistoryCapturePages"};
    // Each run of consecutive pages captures in one trie descent.
    ForEachIndexRun(pages, [&](size_t first, size_t count) {
        Tracked->Write(uint64_t(pages[first]) * HistoryPageBytes, uint64_t(count) * HistoryPageBytes);
    });
}

void Buffer::CaptureWriteElements(std::span<const uint32_t> elements, uint32_t stride) const {
    if (elements.empty() || !stride || !Tracked) return;
    std::vector<uint32_t> sorted;
    if (!std::ranges::is_sorted(elements)) {
        sorted.assign(elements.begin(), elements.end());
        std::ranges::sort(sorted);
        elements = sorted;
    }
    std::vector<uint32_t> pages;
    pages.reserve(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
        const auto element = elements[i];
        if (i && element == elements[i - 1u]) continue;
        const auto first = uint64_t(element) * stride, end = first + stride;
        if (!Storage || end > Storage->ResidentBytes) throw std::out_of_range("Write elements must be resident.");
        for (auto page = first / HistoryPageBytes; page <= (end - 1u) / HistoryPageBytes; ++page)
            if (pages.empty() || pages.back() != page) pages.push_back(uint32_t(page));
    }
    CaptureWritePages(pages);
}

void Buffer::CaptureWriteRanges(std::span<const Range> ranges, uint32_t stride) const {
    if (ranges.empty() || !stride || !Tracked) return;
    std::vector<uint32_t> pages;
    for (const auto range : ranges) {
        if (!range.Count) continue;
        const auto first = uint64_t(range.Offset) * stride;
        const auto end = (uint64_t(range.Offset) + range.Count) * stride;
        if (!Storage || end > Storage->ResidentBytes) throw std::out_of_range("Write ranges must be resident.");
        for (auto page = first / HistoryPageBytes; page <= (end - 1u) / HistoryPageBytes; ++page)
            if (pages.empty() || pages.back() != page) pages.push_back(uint32_t(page));
    }
    if (!std::ranges::is_sorted(pages)) std::ranges::sort(pages);
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    CaptureWritePages(pages);
}

void Buffer::SetUsedSize(uint64_t size) {
    if (!size) {
        for (auto &previous : PreviousWorkspaces) Ctx.Retired.push_back({.Workspace = std::move(previous)});
        PreviousWorkspaces.clear();
    }
    Reserve(size);
    if (Tracked && size != UsedSize) {
        const auto first = std::min(size, UsedSize);
        const auto count = std::max(size, UsedSize) - first;
        CaptureWrite(first, count);
        if (size < UsedSize) std::memset(Contents().data() + size, 0, count);
    }
    UsedSize = size;
}

void Buffer::Track(store::History &history, std::string name) {
    if (Lifetime == BufferLifetime::Workspace) throw std::logic_error("A compute workspace cannot own document history.");
    Tracked = std::make_unique<store::Pages>(uint32_t(HistoryPageBytes), 5, this, [](void *backing, uint64_t bytes) {
        auto &buffer = *static_cast<Buffer *>(backing);
        buffer.Reserve(bytes);
        return buffer.Contents(); }, UsedSize);
    Tracked->Storage = Contents();
    Tracked->Grow(UsedSize);
    if (const auto bytes = Contents(); bytes.size() > UsedSize) std::ranges::fill(bytes.subspan(UsedSize), std::byte{});
    Tracked->Trie.CollectChanged = true;
    history.Track(*Tracked, std::move(name), 0);
    Tracked->Trie.MarkDirty(0, Tracked->Trie.SlotsFor(UsedSize));
}

std::vector<Buffer> CloneFootprints(BufferContext &ctx, std::span<const BufferFootprint> footprints) {
    const profile::CpuScope scope{"CloneFootprints"};
    constexpr uint64_t HistoryPagesPerPhysical = PhysicalPageBytes / HistoryPageBytes;
    std::vector<Buffer> clones;
    std::vector<std::vector<uint32_t>> physical(footprints.size());
    uint64_t count = 0u;
    for (size_t f = 0u; f < footprints.size(); ++f) {
        const auto &[source, pages] = footprints[f];
        if (!source->Storage || source->Lifetime != BufferLifetime::Canonical) throw std::logic_error("Only canonical storage has page clones.");
        const auto resident = source->Storage->ResidentBytes / HistoryPageBytes;
        if (pages.empty()) throw std::invalid_argument("A page clone needs a footprint.");
        for (size_t i = 0u; i < pages.size(); ++i)
            if (pages[i] >= resident || (i && pages[i] <= pages[i - 1u])) throw std::invalid_argument("Clone pages must be increasing, unique, and resident.");
        for (const auto page : pages)
            if (const auto at = uint32_t(page / HistoryPagesPerPhysical); physical[f].empty() || physical[f].back() != at) physical[f].push_back(at);
        count += physical[f].size();
    }
    if (!count) return clones;
    const auto copies = ctx.Ctx.Pages().Allocate(uint32_t(count), false);
    clones.reserve(footprints.size());
    for (size_t f = 0u, first = 0u; f < footprints.size(); ++f) {
        const auto &[source, pages] = footprints[f];
        const auto *bytes = source->Contents().data();
        for (size_t i = 0u, p = 0u; i < pages.size(); ++i) {
            while (physical[f][p] != pages[i] / HistoryPagesPerPhysical) ++p;
            const auto offset = pages[i] % HistoryPagesPerPhysical * HistoryPageBytes;
            std::memcpy(copies[first + p]->Contents().data() + offset, bytes + uint64_t(pages[i]) * HistoryPageBytes, HistoryPageBytes);
        }
        auto &clone = clones.emplace_back(ctx, 0u, source->Slot == InvalidSlot ? SlotType::Buffer : source->Type);
        clone.Storage = std::make_shared<SparseBuffer>(ctx.Ctx, source->Storage->ResidentBytes, physical[f], std::span{copies}.subspan(first, physical[f].size()));
        clone.UpdateSlot();
        first += physical[f].size();
    }
    return clones;
}
} // namespace mtl
