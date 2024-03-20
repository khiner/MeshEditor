#pragma once

#include <vulkan/vulkan.hpp>

#include "vk_mem_alloc.h"

struct VmaBuffer {
    const VmaAllocator &Allocator;
    VkBuffer Buffer{VK_NULL_HANDLE};
    VmaAllocation Allocation{VK_NULL_HANDLE};
    VmaAllocationInfo AllocationInfo{};

    VmaBuffer(const VmaAllocator &allocator, vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memory_usage)
        : Allocator(allocator) {
        vk::BufferCreateInfo buffer_info{{}, size, vk::BufferUsageFlagBits::eTransferSrc | usage, vk::SharingMode::eExclusive};

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = memory_usage;

        VkBufferCreateInfo buffer_create_info = buffer_info;
        if (vmaCreateBuffer(allocator, &buffer_create_info, &alloc_info, &Buffer, &Allocation, nullptr) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create VMA buffer.");
        }
        vmaGetAllocationInfo(allocator, Allocation, &AllocationInfo);
    }
    VmaBuffer(VmaBuffer &&other) noexcept : Allocator(other.Allocator), Buffer(other.Buffer), Allocation(other.Allocation), AllocationInfo(other.AllocationInfo) {
        other.Buffer = VK_NULL_HANDLE;
        other.Allocation = VK_NULL_HANDLE;
    }
    ~VmaBuffer() {
        if (Buffer != VK_NULL_HANDLE) vmaDestroyBuffer(Allocator, Buffer, Allocation);
    }

    VmaBuffer &operator=(VmaBuffer &&other) noexcept {
        if (this != &other) {
            // Free resources
            if (Buffer != VK_NULL_HANDLE) vmaDestroyBuffer(Allocator, Buffer, Allocation);

            // Transfer resources
            Buffer = other.Buffer;
            Allocation = other.Allocation;
            AllocationInfo = other.AllocationInfo;

            // Prevent double-free
            other.Buffer = VK_NULL_HANDLE;
            other.Allocation = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkBuffer operator*() const { return Buffer; }
    VkBuffer Get() const { return Buffer; }

    vk::DeviceSize GetAllocatedSize() const { return AllocationInfo.size; }

    void Update(const void *data, vk::DeviceSize offset, vk::DeviceSize bytes) {
        void *mapped_data = nullptr;
        if (vmaMapMemory(Allocator, Allocation, &mapped_data) == VK_SUCCESS) {
            memcpy(static_cast<char *>(mapped_data) + offset, data, size_t(bytes));
            vmaUnmapMemory(Allocator, Allocation);
        } else {
            throw std::runtime_error("Failed to map VMA buffer memory.");
        }
    }
};

// See https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/quick_start.html
struct VulkanBufferAllocator {
    VmaVulkanFunctions VulkanFunctions{};
    VmaAllocatorCreateInfo AllocatorCreateInfo{};
    VmaAllocator Allocator{};

    VulkanBufferAllocator(vk::PhysicalDevice physical_device, vk::Device device, VkInstance instance) {
        // VulkanFunctions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
        // VulkanFunctions.vkGetDeviceProcAddr = &vkGetDeviceProcAddr;

        // AllocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        // AllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        AllocatorCreateInfo.physicalDevice = physical_device;
        AllocatorCreateInfo.device = device;
        AllocatorCreateInfo.instance = instance;
        AllocatorCreateInfo.pVulkanFunctions = &VulkanFunctions;

        vmaCreateAllocator(&AllocatorCreateInfo, &Allocator);
    }

    ~VulkanBufferAllocator() {
        vmaDestroyAllocator(Allocator);
    }

    VmaBuffer CreateBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, VmaMemoryUsage memory_usage) {
        return {Allocator, size, usage, memory_usage};
    }
};
