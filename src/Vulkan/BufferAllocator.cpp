#include "BufferAllocator.h"
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

    VmaBuffer &operator=(VmaBuffer &&) = delete;

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
void LoggingVmaAllocate(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Allocating {} bytes of memory of type {}", size, memory_type);
}
void LoggingVmaFree(VmaAllocator, uint32_t memory_type, VkDeviceMemory, VkDeviceSize size, void *) {
    std::println("Freeing {} bytes of memory of type {}", size, memory_type);
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
    VmaDeviceMemoryCallbacks memory_callbacks{LoggingVmaAllocate, LoggingVmaFree, nullptr};
    create_info.pDeviceMemoryCallbacks = &memory_callbacks;
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

std::span<std::byte> BufferAllocator::GetMappedData(vk::Buffer b) const {
    auto &info = GetBufferInfo(Vma, b);
    return {static_cast<std::byte *>(info.pMappedData), info.size};
}
std::span<const std::byte> BufferAllocator::GetData(vk::Buffer b) const {
    const auto &info = GetBufferInfo(Vma, b);
    return {static_cast<const std::byte *>(info.pMappedData), info.size};
}
vk::DeviceSize BufferAllocator::GetAllocatedSize(vk::Buffer b) const { return GetBufferInfo(Vma, b).size; }

void BufferAllocator::Write(vk::Buffer b, std::span<const std::byte> src, vk::DeviceSize offset) const {
    if (src.empty() || offset >= GetAllocatedSize(b)) return;

    std::copy(src.begin(), src.end(), GetMappedData(b).subspan(offset).data());
}

void BufferAllocator::Move(vk::Buffer b, vk::DeviceSize from, vk::DeviceSize to, vk::DeviceSize size) const {
    const auto allocated_size = GetAllocatedSize(b);
    if (size == 0 || from + size > allocated_size || to + size > allocated_size) return;

    auto mapped_data = GetMappedData(b);
    std::memmove(mapped_data.subspan(to).data(), mapped_data.subspan(from).data(), size_t(size));
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

void BufferAllocator::Destroy(vk::Buffer buffer) const {
    BuffersByAllocator.at(Vma).erase(static_cast<VkBuffer>(buffer));
}

} // namespace mvk
