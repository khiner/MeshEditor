#include "VulkanContext.h"

#include <format>
#include <iostream>
#include <numeric>
#include <ranges>

#include "VulkanBuffer.h"

VkBool32 DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data
) {
    (void)user_data; // Unused

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
    const vk::ApplicationInfo app{"", {}, "", {}, VK_API_VERSION_1_3};
    Instance = vk::createInstanceUnique({flags, &app, validation_layers, extensions});

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
        nullptr,
        vk::DispatchLoaderDynamic{Instance.get(), vkGetInstanceProcAddr}
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

    // Create logical device (with one queue).
    static const std::vector<const char *> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    static const std::array queue_priority{1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions, &device_features});

    BufferAllocator = std::make_unique<VulkanBufferAllocator>(PhysicalDevice, *Device, *Instance);

    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::vector pool_sizes{
        // Image samplers:
        // 1) The (2D filled shape) silhouette of the selected mesh.
        // 2) The (2D line) silhouette of the selected mesh, after edge detection.
        // 3) The final scene texture sampler.
        // 4) ImGui fonts.
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 4},
        // All uniform buffer descriptors used across all shaders.
        {vk::DescriptorType::eUniformBuffer, 7},
    };
    const uint max_sets = std::accumulate(
        pool_sizes.begin(), pool_sizes.end(), 0u,
        [](uint sum, const vk::DescriptorPoolSize &pool_size) { return sum + pool_size.descriptorCount; }
    );
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
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

VulkanBuffer VulkanContext::CreateBuffer(vk::BufferUsageFlags usage, vk::DeviceSize bytes) const {
    return {
        usage, bytes,
        // Host buffer: host-visible and coherent staging buffer for CPU writes.
        BufferAllocator->CreateBuffer(bytes, vk::BufferUsageFlagBits::eTransferSrc, MemoryUsage::CpuOnly),
        // Device buffer: device-local for fast GPU access.
        BufferAllocator->CreateBuffer(bytes, vk::BufferUsageFlagBits::eTransferDst | usage, MemoryUsage::GpuOnly)
    };
}

// Adapted from https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 for 64-bits.
uint64_t NextPowerOfTwo(uint64_t x) {
    if (x == 0) return 1;

    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

void VulkanContext::UpdateBuffer(VulkanBuffer &buffer, const void *data, vk::DeviceSize offset, vk::DeviceSize bytes) const {
    if (bytes == 0) bytes = buffer.Size;

    const auto &cb = *TransferCommandBuffers[0];

    // Note: `buffer.Size` is the _used_ size, not the allocated size.
    const vk::DeviceSize required_bytes = offset + bytes;
    if (required_bytes > buffer.GetAllocatedSize()) {
        // Create a new buffer with the first large enough power of two.
        // Copy the old device buffer into its device buffer, and replace the old buffer.
        const vk::DeviceSize new_bytes = NextPowerOfTwo(required_bytes);
        VulkanBuffer new_buffer = CreateBuffer(buffer.Usage, new_bytes);
        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(*buffer.DeviceBuffer, *new_buffer.DeviceBuffer, vk::BufferCopy{0, 0, buffer.Size});
        cb.end();

        SubmitTransfer();
        buffer = std::move(new_buffer);
        buffer.Size = required_bytes; // `buffer.Size` is the newly allocated size, so we may need to shrink it.
    } else {
        buffer.Size = std::max(buffer.Size, required_bytes);
    }

    // Copy data to the host buffer.
    buffer.HostBuffer.Update(data, offset, bytes);

    // Copy data from the staging buffer to the device buffer.
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb.copyBuffer(*buffer.HostBuffer, *buffer.DeviceBuffer, vk::BufferCopy{offset, offset, bytes});
    cb.end();

    SubmitTransfer();
}

void VulkanContext::EraseBufferRegion(VulkanBuffer &buffer, vk::DeviceSize offset, vk::DeviceSize bytes) const {
    if (bytes == 0 || offset + bytes > buffer.Size) return;

    if (const vk::DeviceSize move_bytes = buffer.Size - (offset + bytes); move_bytes > 0) {
        const auto &cb = *TransferCommandBuffers[0];
        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(*buffer.DeviceBuffer, *buffer.DeviceBuffer, vk::BufferCopy{offset + bytes, offset, move_bytes});
        cb.end();

        SubmitTransfer();
    }
    buffer.Size -= bytes;
}

// TODO Use separate fence/semaphores for buffer updates and rendering?
void VulkanContext::SubmitTransfer() const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*TransferCommandBuffers[0]);
    Queue.submit(submit, *RenderFence);

    auto wait_result = Device->waitForFences(*RenderFence, VK_TRUE, UINT64_MAX);
    if (wait_result != vk::Result::eSuccess) throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));

    Device->resetFences(*RenderFence);
}