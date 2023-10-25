#pragma once

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.hpp>

#include "Log.h"

#ifdef DEBUG
#define VKB_DEBUG
#endif

inline static void CheckVk(VkResult err) {
    if (err != 0) throw std::runtime_error(std::format("Vulkan error: {}", int(err)));
}

#if defined(VKB_DEBUG)
static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT type, uint64_t object, size_t location, int32_t message_code, const char *layer_prefix, const char *message, void *user_data) {
    if (flags & VK_DEBUG_REPORT_ERROR_BIT_EXT) Log::Error(std::format("Validation Layer: Error: {}: {}", layer_prefix, message));
    else if (flags & VK_DEBUG_REPORT_WARNING_BIT_EXT) Log::Error(std::format("Validation Layer: Warning: {}: {}", layer_prefix, message));
    else if (flags & VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT) Log::Info(std::format("Validation Layer: Performance warning: {}: {}", layer_prefix, message));
    else Log::Info(std::format("Validation Layer: Information: {}: {}", layer_prefix, message));
    return VK_FALSE;
}
#endif

inline static bool IsExtensionAvailable(const std::vector<vk::ExtensionProperties> &properties, const char *extension) {
    for (const vk::ExtensionProperties &p : properties)
        if (strcmp(p.extensionName, extension) == 0)
            return true;
    return false;
}

struct VulkanContext {
    vk::Instance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::Device Device;
    uint32_t QueueFamily = (uint32_t)-1;
    vk::Queue Queue;
    vk::PipelineCache PipelineCache;
    vk::DescriptorPool DescriptorPool;

    void Init(std::vector<const char *> extensions);
    void Uninit();

    // Find a discrete GPU, or the first available (integrated) GPU.
    vk::PhysicalDevice FindPhysicalDevice() const;
};
