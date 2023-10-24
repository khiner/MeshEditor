#pragma once

#include <vulkan/vulkan.h>

struct VulkanContext {
    VkInstance Instance = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkDevice Device = VK_NULL_HANDLE;
    uint32_t QueueFamily = (uint32_t)-1;
    VkQueue Queue = VK_NULL_HANDLE;
    VkPipelineCache PipelineCache = VK_NULL_HANDLE;
    VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
    VkAllocationCallbacks *Allocator = nullptr;
};
