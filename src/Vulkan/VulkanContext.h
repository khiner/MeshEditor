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

    // todo next up: Split `CreateOrUpdateBuffer` into `CreateBuffer` and `UpdateBuffer`, (be explicit about buffer lifecycles).
    //   Then, we can work on deleting stuff from `MeshVkData` (starting with `Models`) and instead update the staging buffers directly.

    // Create the staging and device buffers and their memory.
    // Assumes `buffer.Size` is set.
    void CreateBuffer(VulkanBuffer &, vk::DeviceSize bytes) const;
    // Uses `buffer.Size` if `bytes` is not set.
    void UpdateBuffer(VulkanBuffer &, const void *data, vk::DeviceSize offset = 0, vk::DeviceSize bytes = 0) const;

    template<typename T> void CreateBuffer(VulkanBuffer &buffer, const std::vector<T> &data) const {
        CreateBuffer(buffer, sizeof(T) * data.size());
        UpdateBuffer(buffer, data);
    }
    template<typename T> void UpdateBuffer(VulkanBuffer &buffer, const std::vector<T> &data) const { UpdateBuffer(buffer, data.data(), 0, buffer.Size); }
};
