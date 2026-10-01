#pragma once

#include "metal/MetalContext.h"
#include "metal/PhysicalPages.h"

namespace MTL {
class Buffer;
class Heap;
class CommandBuffer;
}

namespace mtl {
// CPU and GPU virtual ranges map the same placement-heap pages.
// The CPU range covers the device's buffer length limit, and the GPU range covers at least FirstGpuAddressBytes.
// Growing past the GPU range replaces it once with one as large as the CPU range, which maps the same pages.
// The replaced address retires after submitted readers finish.
inline constexpr uint64_t FirstGpuAddressBytes = 1ull << 30;
struct SparseBuffer {
    SparseBuffer(const Context &, uint64_t virtual_bytes);
    // A read-only clone maps each page at its increasing logical index and has no CPU view.
    // Its other pages stay unmapped.
    SparseBuffer(const Context &, uint64_t virtual_bytes, std::span<const uint32_t> indices, std::span<const PhysicalPageRef> pages);
    ~SparseBuffer();
    SparseBuffer(const SparseBuffer &) = delete;
    SparseBuffer &operator=(const SparseBuffer &) = delete;

    void Reserve(uint64_t bytes);
    PhysicalPageRef RetainPage(uint64_t index) const;
    std::span<std::byte> Contents() const { return {reinterpret_cast<std::byte *>(CpuAddress), ResidentBytes}; }

    const Context &Ctx;
    NS::SharedPtr<MTL::Buffer> Gpu;
    uint64_t VirtualBytes{}, CpuAddress{}, ResidentBytes{}, CpuVirtualBytes{};

private:
    std::vector<PhysicalPageRef> Pages;
    std::vector<uint32_t> CloneIndices; // The logical page of each clone page
    void ReserveAddresses(uint64_t bytes);
    void MapCpu(uint64_t first, std::span<const PhysicalPageRef>);
    void MapGpu(uint64_t first, std::span<const PhysicalPageRef>);
};
} // namespace mtl
