#include "VulkanBuffer.h"
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <print>

struct VmaBuffer::AllocationInfo {
    VmaAllocation Allocation{nullptr};
    VmaAllocationInfo Info{};
};

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

#ifndef RELEASE_BUILD
void LoggingVmaAllocate(VmaAllocator, uint32_t memoryType, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Allocating {} bytes of memory of type {}", size, memoryType);
}
void LoggingVmaFree(VmaAllocator, uint32_t memoryType, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Freeing {} bytes of memory of type {}", size, memoryType);
}
#endif
} // namespace

VmaBuffer::VmaBuffer(const VmaAllocator &allocator, vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryUsage memory_usage)
    : Allocator(allocator),
      Allocation(std::make_unique<AllocationInfo>()) {
    VmaAllocationCreateInfo aci{};
    aci.usage = ToVmaMemoryUsage(memory_usage);
    if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
        aci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    vk::BufferCreateInfo bci{{}, size, vk::BufferUsageFlagBits::eTransferSrc | usage, vk::SharingMode::eExclusive};
    if (vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&Buffer), &Allocation->Allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer failed");
    }
    vmaGetAllocationInfo(allocator, Allocation->Allocation, &Allocation->Info);
}

VmaBuffer::VmaBuffer(const VmaAllocator &allocator, VmaAllocation allocation, const VmaAllocationInfo &info, vk::Buffer buffer)
    : Allocator(allocator),
      Allocation(std::make_unique<AllocationInfo>(allocation, info)),
      Buffer(buffer) {}

VmaBuffer::VmaBuffer(VmaBuffer &&other) noexcept
    : Allocator(other.Allocator),
      Allocation(std::move(other.Allocation)),
      Buffer(std::exchange(other.Buffer, nullptr)) {}

VmaBuffer &VmaBuffer::operator=(VmaBuffer &&other) noexcept {
    if (this != &other) {
        if (Buffer) {
            vmaDestroyBuffer(Allocator, Buffer, Allocation->Allocation);
        }
        Buffer = std::exchange(other.Buffer, nullptr);
        Allocation = std::move(other.Allocation);
    }
    return *this;
}

VmaBuffer::~VmaBuffer() {
    if (Buffer) {
        vmaDestroyBuffer(Allocator, Buffer, Allocation->Allocation);
    }
}

const void *VmaBuffer::GetData() const { return Allocation->Info.pMappedData; }
void *VmaBuffer::GetMappedData() { return Allocation->Info.pMappedData; }
vk::DeviceSize VmaBuffer::GetAllocatedSize() const { return Allocation->Info.size; }

void VmaBuffer::WriteRegion(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes) {
    if (bytes == 0 || offset >= GetAllocatedSize()) return;
    memcpy(reinterpret_cast<char *>(GetMappedData()) + offset, data, size_t(bytes));
}

void VmaBuffer::MoveRegion(vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize bytes) {
    if (bytes == 0 || from + bytes > GetAllocatedSize() || to + bytes > GetAllocatedSize()) return;

    // Shift the data to "erase" the region (dst is first, src is second).
    auto *mapped_data = reinterpret_cast<char *>(GetMappedData());
    memmove(mapped_data + to, mapped_data + from, size_t(bytes));
}

struct VulkanBufferAllocator::AllocatorInfo {
    VmaVulkanFunctions VulkanFunctions{};
    VmaAllocatorCreateInfo CreateInfo{};
};

VulkanBufferAllocator::VulkanBufferAllocator(vk::PhysicalDevice physical, vk::Device device, VkInstance instance)
    : Info(std::make_unique<AllocatorInfo>()) {
    Info->CreateInfo.physicalDevice = physical;
    Info->CreateInfo.device = device;
    Info->CreateInfo.instance = instance;
    Info->CreateInfo.pVulkanFunctions = &Info->VulkanFunctions;
#ifndef RELEASE_BUILD
    VmaDeviceMemoryCallbacks memoryCallbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    Info->CreateInfo.pDeviceMemoryCallbacks = &memoryCallbacks;
#endif

    if (vmaCreateAllocator(&Info->CreateInfo, &Allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator.");
    }
}

VulkanBufferAllocator::~VulkanBufferAllocator() {
    vmaDestroyAllocator(Allocator);
}

VmaBuffer VulkanBufferAllocator::CreateVmaBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryUsage memory_usage) const {
    VmaAllocationCreateInfo aci{};
    aci.usage = ToVmaMemoryUsage(memory_usage);
    vk::BufferCreateInfo bci{{}, size, usage, vk::SharingMode::eExclusive};
    if (memory_usage == MemoryUsage::CpuOnly || memory_usage == MemoryUsage::CpuToGpu) {
        aci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
        bci.usage |= vk::BufferUsageFlagBits::eTransferSrc;
    } else {
        bci.usage |= vk::BufferUsageFlagBits::eTransferDst;
    }

    vk::Buffer vk_buffer;
    VmaAllocation alloc;
    VmaAllocationInfo info;
    if (vmaCreateBuffer(Allocator, reinterpret_cast<const VkBufferCreateInfo *>(&bci), &aci, reinterpret_cast<VkBuffer *>(&vk_buffer), &alloc, &info) != VK_SUCCESS) {
        throw std::runtime_error("vmaCreateBuffer in CreateVmaBuffer");
    }

    return {Allocator, alloc, info, vk_buffer};
}
