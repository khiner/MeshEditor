#include "VmaBuffer.h"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

struct VmaBuffer::AllocationInfo {
    VmaAllocation Allocation{VK_NULL_HANDLE};
    VmaAllocationInfo Info{};
};

VmaMemoryUsage ToVmaMemoryUsage(MemoryUsage usage) {
    switch (usage) {
        case MemoryUsage::Unknown: return VMA_MEMORY_USAGE_UNKNOWN;
        case MemoryUsage::GpuOnly: return VMA_MEMORY_USAGE_GPU_ONLY;
        case MemoryUsage::CpuOnly: return VMA_MEMORY_USAGE_CPU_ONLY;
        case MemoryUsage::CpuToGpu: return VMA_MEMORY_USAGE_CPU_TO_GPU;
        case MemoryUsage::GpuToCpu: return VMA_MEMORY_USAGE_GPU_TO_CPU;
        default: throw std::runtime_error("Invalid VMA memory usage.");
    }
}

VmaBuffer::VmaBuffer(const VmaAllocator &allocator, vk::DeviceSize size, vk::BufferUsageFlags usage, MemoryUsage memory_usage)
    : Allocator(allocator), Allocation(std::make_unique<AllocationInfo>()) {
    vk::BufferCreateInfo buffer_info{{}, size, vk::BufferUsageFlagBits::eTransferSrc | usage, vk::SharingMode::eExclusive};

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.usage = ToVmaMemoryUsage(memory_usage);

    VkBufferCreateInfo buffer_create_info = buffer_info;
    if (vmaCreateBuffer(allocator, &buffer_create_info, &alloc_info, &Buffer, &Allocation->Allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA buffer.");
    }
    vmaGetAllocationInfo(allocator, Allocation->Allocation, &Allocation->Info);
}
VmaBuffer::VmaBuffer(VmaBuffer &&other) : Allocator(other.Allocator), Buffer(other.Buffer), Allocation(std::move(other.Allocation)) {
    other.Buffer = VK_NULL_HANDLE;
    other.Allocation = {};
}
VmaBuffer::~VmaBuffer() {
    if (Buffer != VK_NULL_HANDLE) vmaDestroyBuffer(Allocator, Buffer, Allocation->Allocation);
}

VmaBuffer &VmaBuffer::operator=(VmaBuffer &&other) noexcept {
    if (this != &other) {
        // Free resources
        if (Buffer != VK_NULL_HANDLE) vmaDestroyBuffer(Allocator, Buffer, Allocation->Allocation);

        // Transfer resources
        Buffer = other.Buffer;
        Allocation = std::move(other.Allocation);

        // Prevent double-free
        other.Buffer = VK_NULL_HANDLE;
        other.Allocation = {};
    }
    return *this;
}

VkBuffer VmaBuffer::operator*() const { return Buffer; }
VkBuffer VmaBuffer::Get() const { return Buffer; }

vk::DeviceSize VmaBuffer::GetAllocatedSize() const { return Allocation->Info.size; }

void VmaBuffer::Update(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes) {
    void *mapped_data = nullptr;
    if (vmaMapMemory(Allocator, Allocation->Allocation, &mapped_data) == VK_SUCCESS) {
        memcpy(static_cast<char *>(mapped_data) + offset, data, size_t(bytes));
        vmaUnmapMemory(Allocator, Allocation->Allocation);
    } else {
        throw std::runtime_error("Failed to map VMA buffer memory.");
    }
}

struct VulkanBufferAllocator::AllocatorInfo {
    VmaVulkanFunctions VulkanFunctions{};
    VmaAllocatorCreateInfo CreateInfo{};
};

VulkanBufferAllocator::VulkanBufferAllocator(vk::PhysicalDevice physical_device, vk::Device device, VkInstance instance)
    : Info(std::make_unique<AllocatorInfo>()) {
    // VulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    // VulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

    // AllocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    // AllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
    Info->CreateInfo.physicalDevice = physical_device;
    Info->CreateInfo.device = device;
    Info->CreateInfo.instance = instance;
    Info->CreateInfo.pVulkanFunctions = &Info->VulkanFunctions;

    vmaCreateAllocator(&Info->CreateInfo, &Allocator);
}
VulkanBufferAllocator::~VulkanBufferAllocator() {
    vmaDestroyAllocator(Allocator);
}
