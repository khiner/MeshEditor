#pragma once

#include <stdexcept>
#include <vector>
#include <vulkan/vulkan.h>

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

inline static bool IsExtensionAvailable(const std::vector<VkExtensionProperties> &properties, const char *extension) {
    for (const VkExtensionProperties &p : properties)
        if (strcmp(p.extensionName, extension) == 0)
            return true;
    return false;
}

struct VulkanContext {
    VkInstance Instance = VK_NULL_HANDLE;
    VkPhysicalDevice PhysicalDevice = VK_NULL_HANDLE;
    VkDevice Device = VK_NULL_HANDLE;
    uint32_t QueueFamily = (uint32_t)-1;
    VkQueue Queue = VK_NULL_HANDLE;
    VkPipelineCache PipelineCache = VK_NULL_HANDLE;
    VkDescriptorPool DescriptorPool = VK_NULL_HANDLE;
    VkAllocationCallbacks *Allocator = nullptr;

    void Init(std::vector<const char *> extensions);
    void Uninit();

    // Find a discrete GPU, or the first available (integrated) GPU.
    VkPhysicalDevice FindPhysicalDevice() const;
};
