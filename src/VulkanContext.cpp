#include "VulkanContext.h"

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
        vk::DebugUtilsMessengerCreateInfoEXT{
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

    const auto queue_family_props = PhysicalDevice.getQueueFamilyProperties();
    QueueFamily = std::distance(
        queue_family_props.begin(),
        std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const auto &qfp) {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        })
    );
    if (QueueFamily == static_cast<uint>(-1)) throw std::runtime_error("No graphics queue family found.");

    // Create logical device (with 1 queue).
    const std::vector<const char *> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    const std::array<float, 1> queue_priority = {1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions});
    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::array<vk::DescriptorPoolSize, 1> pool_sizes = {
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 2},
    };
    DescriptorPool = Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 2, pool_sizes});
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
