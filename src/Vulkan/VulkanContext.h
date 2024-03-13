#pragma once

#include <vector>

#include "VulkanBuffer.h"

struct VulkanBuffer;

using uint = u_int32_t;

VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT *, void *);

bool IsExtensionAvailable(const std::vector<vk::ExtensionProperties> &, const char *extension);

struct VulkanContext {
    VulkanContext(std::vector<const char *> extensions);
    ~VulkanContext() = default; // Using unique handles, so no need to manually destroy anything.

    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;
    uint QueueFamily{uint(-1)};
    vk::Queue Queue;
    vk::UniquePipelineCache PipelineCache;
    vk::UniqueDescriptorPool DescriptorPool;

    const uint FramebufferCount{1};
    vk::UniqueCommandPool CommandPool;
    std::vector<vk::UniqueCommandBuffer> CommandBuffers;
    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
    vk::UniqueFence RenderFence;

    // Find a discrete GPU, or the first available (integrated) GPU.
    vk::PhysicalDevice FindPhysicalDevice() const;
    uint FindMemoryType(uint type_filter, vk::MemoryPropertyFlags) const;

    // `bytes == 0` defaults to `buffer.Size`.
    // If the buffer doesn't exist, it will be created, _always_ using the buffer size (even if `offset`/`bytes` is specified).
    void CreateOrUpdateBuffer(VulkanBuffer &, const void *data, vk::DeviceSize offset = 0, vk::DeviceSize bytes = 0) const;

    template<typename T> void CreateOrUpdateBuffer(VulkanBuffer &buffer, const std::vector<T> &data) const {
        buffer.Size = sizeof(T) * data.size();
        CreateOrUpdateBuffer(buffer, data.data(), 0, buffer.Size);
    }
};
