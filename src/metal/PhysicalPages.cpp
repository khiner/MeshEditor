#include "metal/PhysicalPages.h"
#include "Parallel.h"
#include "metal/AutoreleaseScope.h"
#include "metal/MetalContext.h"
#include "metal/MetalCpp.h"

#include <algorithm>

namespace mtl {
struct PhysicalPageSlab : std::enable_shared_from_this<PhysicalPageSlab> {
    std::shared_ptr<PhysicalPagePool> Owner;
    NS::SharedPtr<MTL::Heap> Heap;
    NS::SharedPtr<MTL::Buffer> Cpu;
    // Two bitmap levels select the lowest free address in constant time.
    // Releasing pages in any order preserves contiguous runs on reuse.
    std::array<uint64_t, PhysicalSlabPages / 64u> FreeBits;
    uint32_t FreeWords{(1u << FreeBits.size()) - 1u};
    uint32_t FreeCount{PhysicalSlabPages}, HighWater{}, CacheIndex{UINT32_MAX};
    PhysicalPageSlab *Previous{}, *Next{};

    explicit PhysicalPageSlab(std::shared_ptr<PhysicalPagePool> owner) : Owner(std::move(owner)) {
        const AutoreleaseScope pool;
        const auto descriptor = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
        descriptor->setType(MTL::HeapTypePlacement);
        descriptor->setStorageMode(MTL::StorageModeShared);
        descriptor->setMaxCompatiblePlacementSparsePageSize(MTL::SparsePageSize256);
        descriptor->setSize(PhysicalSlabBytes);
        auto heap = NS::TransferPtr(Owner->Ctx.Device->newHeap(descriptor.get()));
        if (!heap) throw std::runtime_error("Failed to allocate canonical Metal pages.");
        auto cpu = NS::TransferPtr(heap->newBuffer(PhysicalSlabBytes, MTL::ResourceStorageModeShared, 0));
        if (!cpu) throw std::runtime_error("Failed to map canonical Metal pages for CPU access.");
        if (reinterpret_cast<uintptr_t>(cpu->contents()) % (16u << 10u)) throw std::runtime_error("Shared heap CPU data is not aligned to a VM page.");
        FreeBits.fill(UINT64_MAX);
        Owner->Ctx.AddResident(heap.get());
        Heap = std::move(heap);
        Cpu = std::move(cpu);
        Owner->HeapBytes += PhysicalSlabBytes;
    }
    ~PhysicalPageSlab() {
        const AutoreleaseScope pool;
        Owner->Ctx.RemoveResident(Heap.get());
        Owner->HeapBytes -= PhysicalSlabBytes;
        AutoreleaseScope::Release(Cpu, Heap);
    }
};

PhysicalPage::PhysicalPage(std::shared_ptr<PhysicalPageSlab> slab, uint32_t index) : Slab(std::move(slab)), Index(index) {}
PhysicalPage::~PhysicalPage() { Slab->Owner->Release(*Slab, Index); }
MTL::Buffer *PhysicalPage::Buffer() const { return Slab->Cpu.get(); }
MTL::Heap *PhysicalPage::Heap() const { return Slab->Heap.get(); }
uint64_t PhysicalPage::Offset() const { return uint64_t(Index) * PhysicalPageBytes; }
std::span<std::byte> PhysicalPage::Contents() const { return {static_cast<std::byte *>(Slab->Cpu->contents()) + Offset(), PhysicalPageBytes}; }

PhysicalPagePool::~PhysicalPagePool() { assert(Available == nullptr && PageCount == 0 && HeapBytes == 0); }

void PhysicalPagePool::AddAvailable(PhysicalPageSlab &slab) {
    slab.Previous = nullptr;
    slab.Next = Available;
    if (Available) Available->Previous = &slab;
    Available = &slab;
}
void PhysicalPagePool::RemoveAvailable(PhysicalPageSlab &slab) {
    if (slab.Previous) slab.Previous->Next = slab.Next;
    else Available = slab.Next;
    if (slab.Next) slab.Next->Previous = slab.Previous;
    slab.Previous = slab.Next = nullptr;
}
void PhysicalPagePool::Release(PhysicalPageSlab &slab, uint32_t index) {
    if (slab.FreeCount == 0) AddAvailable(slab);
    const auto word = index / 64u;
    const auto bit = uint64_t{1} << (index % 64u);
    assert((slab.FreeBits[word] & bit) == 0u);
    slab.FreeBits[word] |= bit;
    slab.FreeWords |= 1u << word;
    ++slab.FreeCount;
    --PageCount;
    if (slab.FreeCount == PhysicalSlabPages) {
        slab.CacheIndex = uint32_t(Cached.size());
        Cached.push_back(slab.shared_from_this());
    }
}

void PhysicalPagePool::TrimCache(uint64_t keep_bytes) {
    const AutoreleaseScope pool;
    while (!Cached.empty() && CachedBytes() > keep_bytes) {
        RemoveAvailable(*Cached.back());
        Cached.pop_back();
    }
}

std::vector<PhysicalPageRef> PhysicalPagePool::Allocate(uint32_t count, bool zeroed) {
    if (uint64_t(count) > Ctx.Device->maxBufferLength() / PhysicalPageBytes) throw std::length_error("Physical page batch exceeds the Metal buffer address space.");
    std::vector<PhysicalPageRef> pages;
    pages.reserve(count);
    std::vector<PhysicalPage *> clear;
    for (uint32_t i = 0; i < count; ++i) {
        // All arenas share these slabs.
        // Reserve enough physical pages to amortize small arena growth and history captures across one heap.
        // Sizing a slab to each request degenerates into one heap per 64 KiB.
        auto slab = Available ? Available->shared_from_this() :
                                std::make_shared<PhysicalPageSlab>(shared_from_this());
        if (!Available) AddAvailable(*slab);
        if (slab->CacheIndex != UINT32_MAX) {
            const auto at = slab->CacheIndex;
            if (at + 1u != Cached.size()) {
                Cached[at] = std::move(Cached.back());
                Cached[at]->CacheIndex = at;
            }
            Cached.pop_back();
            slab->CacheIndex = UINT32_MAX;
        }
        const auto word = std::countr_zero(slab->FreeWords);
        const auto index = uint32_t(word * 64u + std::countr_zero(slab->FreeBits[word]));
        auto page = PhysicalPageRef(new PhysicalPage(slab, index));
        slab->FreeBits[word] &= slab->FreeBits[word] - 1u;
        if (!slab->FreeBits[word]) slab->FreeWords &= ~(1u << word);
        if (--slab->FreeCount == 0) RemoveAvailable(*slab);
        ++PageCount;
        if (index < slab->HighWater) {
            ++ReuseCount;
            if (zeroed) clear.push_back(page.get());
        }
        slab->HighWater = std::max(slab->HighWater, index + 1u);
        pages.push_back(std::move(page));
    }
    // A recycled page has no pending GPU readers or writers, so the CPU clears it through its unified-memory view.
    ParallelFor(uint32_t(clear.size()), [&](uint32_t i) { std::ranges::fill(clear[i]->Contents(), std::byte{}); });
    return pages;
}
} // namespace mtl
