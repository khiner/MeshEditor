#include "Buffer.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <print>
#include <unordered_map>

namespace {
struct VmaBuffer {
    VmaBuffer(VmaAllocator vma, vk::Buffer buffer, VmaAllocation allocation, VmaAllocationInfo info)
        : Vma(vma),
          Buffer(buffer),
          Allocation(allocation),
          Info(info) {}

    VmaBuffer(VmaBuffer &&other) noexcept
        : Vma(other.Vma),
          Buffer(std::exchange(other.Buffer, nullptr)),
          Allocation(std::exchange(other.Allocation, nullptr)),
          Info(other.Info) {}

    VmaBuffer &operator=(VmaBuffer &&other) noexcept {
        if (this != &other) {
            if (Buffer) {
                vmaDestroyBuffer(Vma, Buffer, Allocation);
            }
            Buffer = std::exchange(other.Buffer, nullptr);
            Allocation = std::move(other.Allocation);
        }
        return *this;
    }

    ~VmaBuffer() {
        if (Buffer) {
            vmaDestroyBuffer(Vma, Buffer, Allocation);
        }
    }

    vk::Buffer operator*() const {
        return Buffer;
    }

    VmaAllocator Vma;
    vk::Buffer Buffer;
    VmaAllocation Allocation{nullptr};
    VmaAllocationInfo Info{};
};

using BufferMap = std::unordered_map<VkBuffer, VmaBuffer>;
std::unordered_map<VmaAllocator, BufferMap> BuffersByAllocator;

VmaBuffer &GetVmaBuffer(VmaAllocator vma, vk::Buffer buffer) {
    return BuffersByAllocator.at(vma).at(static_cast<VkBuffer>(buffer));
}
// vk::Buffer GetBuffer(VmaAllocator vma, vk::Buffer buffer) {
//     const auto &vma_buffer = GetVmaBuffer(vma, buffer);
//     return vma_buffer.Buffer;
// }
const VmaAllocationInfo &GetBufferInfo(VmaAllocator vma, vk::Buffer buffer) {
    return GetVmaBuffer(vma, buffer).Info;
}

vk::Buffer SetBuffer(VmaAllocator vma, VmaBuffer buffer) {
    auto vk_buffer = *buffer;
    BuffersByAllocator.at(vma).emplace(static_cast<VkBuffer>(vk_buffer), std::move(buffer));
    return vk_buffer;
}

struct AllocatorInfo {
    VmaVulkanFunctions VulkanFunctions{};
};

#ifndef RELEASE_BUILD
void LoggingVmaAllocate(VmaAllocator, uint32_t memoryType, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Allocating {} bytes of memory of type {}", size, memoryType);
}
void LoggingVmaFree(VmaAllocator, uint32_t memoryType, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Freeing {} bytes of memory of type {}", size, memoryType);
}
#endif
} // namespace

// vmaDestroyBuffer(Vma, Buffer, Allocation->Allocation);
namespace mvk {
BufferAllocator::BufferAllocator(vk::PhysicalDevice pd, vk::Device d, VkInstance instance) {
    VmaAllocatorCreateInfo create_info{};
    create_info.physicalDevice = pd;
    create_info.device = d;
    create_info.instance = instance;
#ifndef RELEASE_BUILD
    VmaDeviceMemoryCallbacks memoryCallbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    create_info.pDeviceMemoryCallbacks = &memoryCallbacks;
#endif

    if (vmaCreateAllocator(&create_info, &Vma) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator.");
    }
    BuffersByAllocator.emplace(Vma, BufferMap{});
}

BufferAllocator::~BufferAllocator() {
    BuffersByAllocator.erase(Vma);
    vmaDestroyAllocator(Vma);
}

void *BufferAllocator::GetMappedData(vk::Buffer b) { return GetBufferInfo(Vma, b).pMappedData; }
const void *BufferAllocator::GetData(vk::Buffer b) const { return GetBufferInfo(Vma, b).pMappedData; }
vk::DeviceSize BufferAllocator::GetAllocatedSize(vk::Buffer b) const { return GetBufferInfo(Vma, b).size; }

void BufferAllocator::WriteRegion(vk::Buffer b, const void *data, vk::DeviceSize offset, vk::DeviceSize bytes) {
    if (bytes == 0 || offset >= GetAllocatedSize(b)) return;
    memcpy(reinterpret_cast<char *>(GetMappedData(b)) + offset, data, size_t(bytes));
}

void BufferAllocator::MoveRegion(vk::Buffer b, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize bytes) {
    if (bytes == 0 || from + bytes > GetAllocatedSize(b) || to + bytes > GetAllocatedSize(b)) return;

    // Shift the data to "erase" the region (dst is first, src is second).
    auto *mapped_data = reinterpret_cast<char *>(GetMappedData(b));
    memmove(mapped_data + to, mapped_data + from, size_t(bytes));
}

namespace {
VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
    }
}
} // namespace

vk::Buffer BufferAllocator::Allocate(vk::DeviceSize size, MemoryUsage memory_usage, vk::BufferUsageFlags usage) const {
    VmaAllocationCreateInfo aci{};
    aci.usage = ToVmaMemoryUsage(memory_usage);
    // All staging and device buffers act as transfer sources, since buffer updates include a device->device copy.
    vk::BufferCreateInfo bci{{}, size, usage | vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
    if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
        aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    } else {
        bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    vk::Buffer vk_buffer;
    VmaAllocation alloc;
    VmaAllocationInfo info;
    if (vmaCreateBuffer(Vma, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&vk_buffer), &alloc, &info) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed");
    }
    return SetBuffer(Vma, {Vma, vk_buffer, alloc, info});
}
} // namespace mvk
