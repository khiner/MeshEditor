#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace MTL { class Buffer; class Heap; }

// Canonical buffers and page clones map physical pages of this size.
inline constexpr uint64_t PhysicalPageBytes = 256u << 10;

namespace mtl {
struct Context;
struct PhysicalPageSlab;

inline constexpr uint32_t PhysicalSlabPages = 64u;
inline constexpr uint64_t PhysicalSlabBytes = PhysicalPageBytes * PhysicalSlabPages;

// A reference owns one physical page independently of any arena mapping.
// Keep the reference until GPU commands using the page have completed.
struct PhysicalPage {
    ~PhysicalPage();
    PhysicalPage(const PhysicalPage &) = delete;
    PhysicalPage &operator=(const PhysicalPage &) = delete;

    MTL::Buffer *Buffer() const;
    MTL::Heap *Heap() const;
    uint64_t Offset() const;
    std::span<std::byte> Contents() const;

private:
    friend struct PhysicalPagePool;
    PhysicalPage(std::shared_ptr<PhysicalPageSlab>, uint32_t index);
    std::shared_ptr<PhysicalPageSlab> Slab;
    uint32_t Index;
};
using PhysicalPageRef = std::shared_ptr<PhysicalPage>;

// The available-slab list is intrusive: allocating/releasing a page never
// searches other slabs. Empty slabs are reused until trimming at a GPU idle point.
struct PhysicalPagePool : std::enable_shared_from_this<PhysicalPagePool> {
    explicit PhysicalPagePool(const Context &ctx) : Ctx(ctx) {}
    ~PhysicalPagePool();
    PhysicalPagePool(const PhysicalPagePool &) = delete;
    PhysicalPagePool &operator=(const PhysicalPagePool &) = delete;

    // Zeroed pages are immediately readable by the CPU and the GPU.
    // Uninitialized pages must be fully written before publication.
    std::vector<PhysicalPageRef> Allocate(uint32_t count, bool zeroed = true);
    uint64_t LivePages() const { return PageCount; }
    uint64_t ResidentBytes() const { return HeapBytes; }
    uint64_t CachedBytes() const { return uint64_t(Cached.size()) * PhysicalSlabBytes; }
    uint64_t ReusedPages() const { return ReuseCount; }
    const Context &Ctx;

private:
    friend struct Context;
    friend struct PhysicalPage;
    friend struct PhysicalPageSlab;
    PhysicalPageSlab *Available{};
    std::vector<std::shared_ptr<PhysicalPageSlab>> Cached;
    void TrimCache(uint64_t keep_bytes);
    uint64_t PageCount{}, HeapBytes{}, ReuseCount{};
    void AddAvailable(PhysicalPageSlab &);
    void RemoveAvailable(PhysicalPageSlab &);
    void Release(PhysicalPageSlab &, uint32_t index);
};
} // namespace mtl
