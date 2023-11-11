#include "VulkanContext.h"

#include <format>
#include <iostream>
#include <numeric>
#include <ranges>

VkBool32 DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data
) {
    (void)user_data; // Unused.

    const char *severity_str = "";
    switch (severity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT: severity_str = "Verbose"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT: severity_str = "Info"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT: severity_str = "Warning"; break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT: severity_str = "Error"; break;
        default: break;
    }

    const char *type_str = "";
    switch (type) {
        case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT: type_str = "General"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT: type_str = "Validation"; break;
        case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT: type_str = "Performance"; break;
        default: break;
    }

    std::cerr << "[Vulkan|" << severity_str << "|" << type_str << "]"
              << ": " << callback_data->pMessage << std::endl;

    return VK_FALSE; // Return VK_TRUE if the message is to be aborted and VK_FALSE otherwise.
}

bool IsExtensionAvailable(const std::vector<vk::ExtensionProperties> &properties, const char *extension) {
    return std::ranges::any_of(properties, [extension](const auto &prop) { return strcmp(prop.extensionName, extension) == 0; });
}

VulkanContext::VulkanContext(std::vector<const char *> extensions) {
    const auto instance_props = vk::enumerateInstanceExtensionProperties();
    if (IsExtensionAvailable(instance_props, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }

    vk::InstanceCreateFlags flags;
    if (IsExtensionAvailable(instance_props, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
    extensions.push_back("VK_EXT_debug_utils");

    const std::vector<const char *> validation_layers{"VK_LAYER_KHRONOS_validation"};
    const vk::ApplicationInfo app{"", {}, "", {}, {}};
    Instance = vk::createInstanceUnique({flags, &app, validation_layers, extensions});

    const vk::DispatchLoaderDynamic dldi{Instance.get(), vkGetInstanceProcAddr};
    const auto messenger = Instance->createDebugUtilsMessengerEXTUnique(
        {
            {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            DebugCallback,
        },
        nullptr, dldi
    );

    PhysicalDevice = FindPhysicalDevice();

    const auto qfp = PhysicalDevice.getQueueFamilyProperties();
    const auto qfp_find_graphics_it = std::ranges::find_if(qfp, [](const auto &qfp) { return bool(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });
    if (qfp_find_graphics_it == qfp.end()) throw std::runtime_error("No graphics queue family found.");

    QueueFamily = std::ranges::distance(qfp.begin(), qfp_find_graphics_it);
    if (!PhysicalDevice.getFeatures().fillModeNonSolid) {
        throw std::runtime_error("`fillModeNonSolid` is not supported, but is needed for line rendering.");
    }

    vk::PhysicalDeviceFeatures device_features{};
    device_features.fillModeNonSolid = VK_TRUE;

    // Create logical device (with 1 queue).
    const std::vector<const char *> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    const std::array queue_priority = {1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions, &device_features});
    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::vector pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 2}, // 1 for ImGui and 1 for the Scene texture sampler.
        {vk::DescriptorType::eUniformBuffer, 3}, // Two shader pipelines use a transform buffer and one uses a light buffer.
    };
    const uint max_sets = std::accumulate(pool_sizes.begin(), pool_sizes.end(), 0u, [](uint sum, const vk::DescriptorPoolSize &pool_size) {
        return sum + pool_size.descriptorCount;
    });
    DescriptorPool = Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_sets, pool_sizes});

    CommandPool = Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, QueueFamily});
    CommandBuffers = Device->allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, FramebufferCount});
    TransferCommandBuffers = Device->allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, FramebufferCount});
    RenderFence = Device->createFenceUnique({});
}

vk::PhysicalDevice VulkanContext::FindPhysicalDevice() const {
    const auto physical_devices = Instance->enumeratePhysicalDevices();
    if (physical_devices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physical_devices) {
        if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }
    return physical_devices[0];
}

uint VulkanContext::FindMemoryType(uint type_filter, vk::MemoryPropertyFlags prop_flags) const {
    auto mem_props = PhysicalDevice.getMemoryProperties();
    for (uint i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanContext::CreateOrUpdateBuffer(VulkanBuffer &buffer, const void *data, bool force_recreate) const {
    // Optionally create the staging buffer and its memory.
    if (force_recreate || !buffer.StagingBuffer || !buffer.StagingMemory) {
        buffer.StagingBuffer = Device->createBufferUnique({{}, buffer.Size, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive});
        const auto staging_mem_reqs = Device->getBufferMemoryRequirements(*buffer.StagingBuffer);
        buffer.StagingMemory = Device->allocateMemoryUnique({staging_mem_reqs.size, FindMemoryType(staging_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)});
        Device->bindBufferMemory(*buffer.StagingBuffer, *buffer.StagingMemory, 0);
    }

    // Copy data to the staging buffer.
    void *mapped_data = Device->mapMemory(*buffer.StagingMemory, 0, buffer.Size);
    memcpy(mapped_data, data, size_t(buffer.Size));
    Device->unmapMemory(*buffer.StagingMemory);

    // Optionally create the device buffer and its memory.
    if (force_recreate || !buffer.Buffer || !buffer.Memory) {
        buffer.Buffer = Device->createBufferUnique({{}, buffer.Size, vk::BufferUsageFlagBits::eTransferDst | buffer.Usage, vk::SharingMode::eExclusive});
        const auto buffer_mem_reqs = Device->getBufferMemoryRequirements(*buffer.Buffer);
        buffer.Memory = Device->allocateMemoryUnique({buffer_mem_reqs.size, FindMemoryType(buffer_mem_reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)});
        Device->bindBufferMemory(*buffer.Buffer, *buffer.Memory, 0);
    }

    // Copy data from the staging buffer to the device buffer.
    const auto &command_buffer = TransferCommandBuffers[0];
    command_buffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    vk::BufferCopy copy_region;
    copy_region.size = buffer.Size;
    command_buffer->copyBuffer(*buffer.StagingBuffer, *buffer.Buffer, copy_region);
    command_buffer->end();

    // TODO we should use a separate fence/semaphores for buffer updates and rendering.
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*command_buffer);
    Queue.submit(submit, *RenderFence);

    auto wait_result = Device->waitForFences(*RenderFence, VK_TRUE, UINT64_MAX);
    if (wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    Device->resetFences(*RenderFence);
}
