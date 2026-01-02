#include "Buffer.h"
#include "../Bindless.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <algorithm>
#include <cassert>
#include <print>
#include <ranges>

using std::views::transform, std::ranges::to;

namespace mvk {
namespace {
uint64_t NextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    }
}

std::string FormatBytes(uint32_t bytes) {
    static constexpr std::array<std::string_view, 6> Suffixes{"B", "KB", "MB", "GB", "TB", "PB"};
    static constexpr float K = 1024;
    auto value = float(bytes);
    size_t pow;
    for (pow = 0; value >= K && pow + 1 < Suffixes.size(); ++pow) value /= K;
    return std::format("{:.2f} {}", value, Suffixes[pow]);
}

#ifndef RELEASE_BUILD
void LoggingVmaAllocate(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Allocating {} bytes of memory of type {}", FormatBytes(size), memory_type);
}
void LoggingVmaFree(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Freeing {} bytes of memory of type {}", FormatBytes(size), memory_type);
}
#endif

std::vector<VmaBudget> QueryHeapBudgets(VmaAllocator allocator, vk::PhysicalDevice pd) {
    VkPhysicalDeviceMemoryProperties memory_props;
    vkGetPhysicalDeviceMemoryProperties(pd, &memory_props);
    std::vector<VmaBudget> budgets(memory_props.memoryHeapCount);
    vmaGetHeapBudgets(allocator, budgets.data());
    return budgets;
}
} // namespace

struct VmaBuffer {
    VmaBuffer(VmaAllocator vma, vk::DeviceSize size, MemoryUsage memory_usage, vk::BufferUsageFlags usage) : Vma(vma) {
        VmaAllocationCreateInfo aci{};
        aci.usage = ToVmaMemoryUsage(memory_usage);
        if (memory_usage == MemoryUsage::GpuOnly) {
#ifdef MVK_FORCE_STAGED_TRANSFERS
            // Force GPU-only memory without host-visible preference to exercise staging path
            aci.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
#else
            // Prefer host-visible for direct writes on unified memory architectures
            aci.usage = VMA_MEMORY_USAGE_AUTO;
            aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;
#endif
        }
        vk::BufferCreateInfo bci{{}, size, usage | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
        if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
            aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        } else {
            bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
        }
        if (vmaCreateBuffer(Vma, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&Handle), &Allocation, &Info) != VK_SUCCESS) {
            throw std::runtime_error("vmaCreateBuffer failed");
        }
        vmaGetMemoryTypeProperties(Vma, Info.memoryType, &MemoryProps);
    }
    VmaBuffer(VmaAllocator vma, std::span<const std::byte> data, MemoryUsage mem, vk::BufferUsageFlags usage)
        : VmaBuffer(vma, data.size(), mem, usage) { Write(data); }
    ~VmaBuffer() { vmaDestroyBuffer(Vma, Handle, Allocation); }

    vk::Buffer Get() const { return Handle; }
    vk::DeviceSize GetAllocatedSize() const { return Info.size; }
    std::span<const std::byte> GetData() const { return {static_cast<const std::byte *>(Info.pMappedData), Info.size}; }
    std::span<std::byte> GetMappedData() const { return {static_cast<std::byte *>(Info.pMappedData), Info.size}; }
    bool IsDirectMapped() const {
        return Info.pMappedData != nullptr &&
            (MemoryProps & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }

    void Write(std::span<const std::byte> src, vk::DeviceSize offset = 0) const {
        if (src.empty() || offset >= GetAllocatedSize()) return;
        std::copy(src.begin(), src.end(), GetMappedData().subspan(offset).data());
    }
    void Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const {
        const auto allocated_size = GetAllocatedSize();
        if (size == 0 || from + size > allocated_size || to + size > allocated_size) return;
        const auto mapped_data = GetMappedData();
        std::memmove(mapped_data.subspan(to).data(), mapped_data.subspan(from).data(), size_t(size));
    }

    VmaAllocator Vma;
    vk::Buffer Handle;
    VmaAllocation Allocation;
    VmaAllocationInfo Info;
    VkMemoryPropertyFlags MemoryProps{};
};

BufferContext::BufferContext(vk::PhysicalDevice pd, vk::Device d, vk::Instance instance, DescriptorSlots &slots)
    : PhysicalDevice(pd), Slots(slots) {
    VmaAllocatorCreateInfo create_info{};
    create_info.physicalDevice = PhysicalDevice;
    create_info.device = d;
    create_info.instance = instance;
#ifndef RELEASE_BUILD
    VmaDeviceMemoryCallbacks memory_callbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    create_info.pDeviceMemoryCallbacks = &memory_callbacks;
#endif
    if (vmaCreateAllocator(&create_info, &Vma) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator.");
    }
}

BufferContext::~BufferContext() {
    Retired.clear();
    vmaDestroyAllocator(Vma);
}

void BufferContext::ReclaimRetiredBuffers() { Retired.clear(); }

std::string BufferContext::DebugHeapUsage() const {
    const auto budgets = QueryHeapBudgets(Vma, PhysicalDevice);
    std::string result;
    for (uint32_t i = 0; i < budgets.size(); ++i) {
        const auto &b = budgets[i];
        result += std::format(
            "Heap {}/{}:\n\tAllocations:\n\t\tCount: {}\n\t\tBytes: {}\n\tBlocks:\n\t\tCount: {}\n\t\tBytes: {}\n\tTotal:\n\t\tUsed: {}\n\t\tBudget: {}\n",
            i + 1, budgets.size(), b.statistics.allocationCount, FormatBytes(b.statistics.allocationBytes),
            b.statistics.blockCount, FormatBytes(b.statistics.blockBytes), FormatBytes(b.usage), FormatBytes(b.budget)
        );
    }
    return result;
}

std::vector<vk::WriteDescriptorSet> BufferContext::GetDeferredDescriptorUpdates() {
    return DeferredDescriptorUpdates | transform([&](const auto &kv) { return Slots.MakeBufferWrite(kv.first, kv.second); }) | to<std::vector>();
}

#ifdef MVK_FORCE_STAGED_TRANSFERS
// Merges into any overlapping/adjacent deferred copy ranges.
void BufferContext::DeferCopy(vk::Buffer src, vk::Buffer dst, vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0) return;

    auto &ranges = DeferredBufferCopies[{src, dst}];
    auto end = offset + size;
    // Find first range starting after our offset - previous range may overlap
    auto it = ranges.upper_bound(offset);
    if (it != ranges.begin()) {
        if (auto prev = std::prev(it); prev->second >= offset) {
            offset = prev->first;
            end = std::max(end, prev->second);
            it = ranges.erase(prev);
        }
    }
    // Merge with any subsequent overlapping/adjacent ranges
    while (it != ranges.end() && it->first <= end) {
        end = std::max(end, it->second);
        it = ranges.erase(it);
    }
    ranges.emplace(offset, end);
}

void BufferContext::CancelDeferredCopies(vk::Buffer src, vk::Buffer dst) {
    DeferredBufferCopies.erase({src, dst});
}
#endif

Buffer::Buffer(BufferContext &ctx, vk::DeviceSize size, vk::BufferUsageFlags usage, SlotType slot_type)
    : Ctx(ctx), Slot(ctx.Slots.Allocate(slot_type)), Usage(usage),
      DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, size, MemoryUsage::GpuOnly, usage | vk::BufferUsageFlagBits::eTransferDst)),
      Type(slot_type) {
#ifdef MVK_FORCE_STAGED_TRANSFERS
    HostBuffer = std::make_unique<VmaBuffer>(Ctx.Vma, size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
#else
    assert(DeviceBuffer->IsDirectMapped() && "Device doesn't support direct-mapped memory. Build with MVK_FORCE_STAGED_TRANSFERS.");
#endif
    UpdateDescriptor();
}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, vk::BufferUsageFlags usage, SlotType slot_type)
    : Buffer(ctx, data.size(), usage, slot_type) { Update(data); }

Buffer::Buffer(BufferContext &ctx, vk::DeviceSize size, MemoryUsage mem, vk::BufferUsageFlags usage)
    : Ctx(ctx), Usage(usage), DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, size, mem, usage)) {}

Buffer::Buffer(BufferContext &ctx, std::span<const std::byte> data, MemoryUsage mem, vk::BufferUsageFlags usage)
    : Ctx(ctx), Usage(usage), DeviceBuffer(std::make_unique<VmaBuffer>(Ctx.Vma, data, mem, usage)) {}

Buffer::Buffer(Buffer &&other)
    : Ctx(other.Ctx), Slot(other.Slot), UsedSize(other.UsedSize), Usage(other.Usage),
      DeviceBuffer(std::move(other.DeviceBuffer)),
#ifdef MVK_FORCE_STAGED_TRANSFERS
      HostBuffer(std::move(other.HostBuffer)),
#endif
      Type(other.Type) {
    other.Slot = InvalidSlot;
}

Buffer &Buffer::operator=(Buffer &&other) {
    if (this != &other) {
        Retire();
        if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
        Slot = other.Slot;
        UsedSize = other.UsedSize;
        Usage = other.Usage;
        DeviceBuffer = std::move(other.DeviceBuffer);
#ifdef MVK_FORCE_STAGED_TRANSFERS
        HostBuffer = std::move(other.HostBuffer);
#endif
        Type = other.Type;
        other.Slot = InvalidSlot;
    }
    return *this;
}

Buffer::~Buffer() {
    Retire();
    if (Slot != InvalidSlot) Ctx.Slots.Release({Type, Slot});
}

void Buffer::Retire() {
    if (!DeviceBuffer) return; // Already moved-from
#ifdef MVK_FORCE_STAGED_TRANSFERS
    if (!HostBuffer) return;
    Ctx.CancelDeferredCopies(HostBuffer->Get(), DeviceBuffer->Get());
    Ctx.Retired.emplace_back(std::move(HostBuffer));
#endif
    Ctx.Retired.emplace_back(std::move(DeviceBuffer));
    if (Slot != InvalidSlot) Ctx.CancelDeferredDescriptorUpdate({Type, Slot});
}

void Buffer::UpdateDescriptor() {
    if (Slot == InvalidSlot) return;
    Ctx.DeferDescriptorUpdate({Type, Slot}, GetDescriptor());
}

vk::Buffer Buffer::operator*() const { return DeviceBuffer->Get(); }
std::span<const std::byte> Buffer::GetData() const { return DeviceBuffer->GetData(); }
vk::DeviceSize Buffer::GetAllocatedSize() const { return DeviceBuffer->GetAllocatedSize(); }

std::span<const std::byte> Buffer::GetMappedData() const {
#ifdef MVK_FORCE_STAGED_TRANSFERS
    return HostBuffer->GetData();
#else
    return DeviceBuffer->GetData();
#endif
}

std::span<std::byte> Buffer::GetMappedData() {
#ifdef MVK_FORCE_STAGED_TRANSFERS
    return HostBuffer->GetMappedData();
#else
    return DeviceBuffer->GetMappedData();
#endif
}

void Buffer::Write(std::span<const std::byte> data, vk::DeviceSize offset) const { DeviceBuffer->Write(data, offset); }
void Buffer::Move(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const { DeviceBuffer->Move(from, to, size); }

std::span<std::byte> Buffer::GetMutableRange(vk::DeviceSize offset, vk::DeviceSize size) {
#ifdef MVK_FORCE_STAGED_TRANSFERS
    // Assume the whole range is modified and schedule a copy
    Ctx.DeferCopy(HostBuffer->Get(), DeviceBuffer->Get(), offset, size);
    return HostBuffer->GetMappedData().subspan(offset, size);
#else
    return DeviceBuffer->GetMappedData().subspan(offset, size);
#endif
}

void Buffer::Reserve(vk::DeviceSize required_size) {
    if (required_size <= DeviceBuffer->GetAllocatedSize()) return;
    const auto new_size = NextPowerOfTwo(required_size);
#ifdef MVK_FORCE_STAGED_TRANSFERS
    auto new_host = std::make_unique<VmaBuffer>(Ctx.Vma, new_size, MemoryUsage::CpuToGpu, vk::BufferUsageFlagBits::eTransferSrc);
    auto new_device = std::make_unique<VmaBuffer>(Ctx.Vma, new_size, MemoryUsage::GpuOnly, Usage | vk::BufferUsageFlagBits::eTransferDst);
    if (UsedSize > 0) {
        new_host->Write(HostBuffer->GetData());
        Ctx.DeferCopy(new_host->Get(), new_device->Get(), 0, UsedSize);
    }
    Ctx.CancelDeferredCopies(HostBuffer->Get(), DeviceBuffer->Get());
    Ctx.Retired.emplace_back(std::move(HostBuffer));
    Ctx.Retired.emplace_back(std::move(DeviceBuffer));
    HostBuffer = std::move(new_host);
    DeviceBuffer = std::move(new_device);
#else
    auto new_device = std::make_unique<VmaBuffer>(Ctx.Vma, new_size, MemoryUsage::GpuOnly, Usage | vk::BufferUsageFlagBits::eTransferDst);
    if (UsedSize > 0) new_device->Write(DeviceBuffer->GetData());
    Ctx.Retired.emplace_back(std::move(DeviceBuffer));
    DeviceBuffer = std::move(new_device);
#endif
    UpdateDescriptor();
}

void Buffer::Update(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty()) return;
    const auto required_size = offset + data.size();
    const auto old_size = DeviceBuffer->GetAllocatedSize();
    Reserve(required_size);
    UsedSize = std::max(UsedSize, required_size);
#ifdef MVK_FORCE_STAGED_TRANSFERS
    HostBuffer->Write(data, offset);
    Ctx.DeferCopy(HostBuffer->Get(), DeviceBuffer->Get(), offset, data.size());
#else
    DeviceBuffer->Write(data, offset);
#endif
    if (DeviceBuffer->GetAllocatedSize() != old_size) UpdateDescriptor();
}

void Buffer::Insert(std::span<const std::byte> data, vk::DeviceSize offset) {
    if (data.empty() || UsedSize + data.size() > DeviceBuffer->GetAllocatedSize()) return;
#ifdef MVK_FORCE_STAGED_TRANSFERS
    if (offset < UsedSize) HostBuffer->Move(offset, offset + data.size(), UsedSize - offset);
    HostBuffer->Write(data, offset);
    UsedSize += data.size();
    Ctx.DeferCopy(HostBuffer->Get(), DeviceBuffer->Get(), offset, UsedSize - offset);
#else
    if (offset < UsedSize) DeviceBuffer->Move(offset, offset + data.size(), UsedSize - offset);
    DeviceBuffer->Write(data, offset);
    UsedSize += data.size();
#endif
}

void Buffer::Erase(vk::DeviceSize offset, vk::DeviceSize size) {
    if (size == 0 || offset + size > UsedSize) return;
    const auto move_size = UsedSize - (offset + size);
#ifdef MVK_FORCE_STAGED_TRANSFERS
    if (move_size > 0) {
        HostBuffer->Move(offset + size, offset, move_size);
        Ctx.DeferCopy(HostBuffer->Get(), DeviceBuffer->Get(), offset, move_size);
    }
#else
    if (move_size > 0) DeviceBuffer->Move(offset + size, offset, move_size);
#endif
    UsedSize -= size;
}
} // namespace mvk
