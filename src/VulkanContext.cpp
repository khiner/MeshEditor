#include "VulkanContext.h"

void VulkanContext::Init(std::vector<const char *> extensions) {
    // Create instance.
    vk::InstanceCreateFlags flags;
    // Enumerate available extensions.
    const auto instance_props = vk::enumerateInstanceExtensionProperties();
    // Enable required extensions.
    if (IsExtensionAvailable(instance_props, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    }
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (IsExtensionAvailable(instance_props, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    }
#endif
    std::vector<const char *> validation_layers;
#ifdef VKB_DEBUG
    validation_layers.push_back("VK_LAYER_KHRONOS_validation");
    extensions.push_back("VK_EXT_debug_report");
#endif

    vk::ApplicationInfo app("", {}, "", {}, {});
    vk::InstanceCreateInfo instance_info{flags, &app, validation_layers, extensions};
    Instance = vk::createInstance(instance_info);

#ifdef VKB_DEBUG
    // Setup the debug report callback.
    auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)instance.getProcAddr("vkCreateDebugReportCallbackEXT");
    IM_ASSERT(vkCreateDebugReportCallbackEXT != nullptr);
    vk::DebugReportCallbackCreateInfoEXT debug_report_ci;
    debug_report_ci.flags = vk::DebugReportFlagBitsEXT::eError | vk::DebugReportFlagBitsEXT::eWarning | vk::DebugReportFlagBitsEXT::ePerformanceWarning;
    debug_report_ci.pfnCallback = debug_report;
    vk::DebugReportCallbackEXT debugReport = instance.createDebugReportCallbackEXT(debug_report_ci);
#endif

    // Select physical device (GPU).
    PhysicalDevice = FindPhysicalDevice();

    // Select graphics queue family.
    const auto queue_family_props = PhysicalDevice.getQueueFamilyProperties();
    QueueFamily = std::distance(
        queue_family_props.begin(),
        std::find_if(queue_family_props.begin(), queue_family_props.end(), [](const auto &qfp) {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        })
    );
    if (QueueFamily == static_cast<uint32_t>(-1)) throw std::runtime_error("No graphics queue family found.");

    // Create logical device (with 1 queue).
    const std::vector<const char *> device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    const std::array<float, 1> queue_priority = {1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    const vk::DeviceCreateInfo device_info({}, queue_info, {}, device_extensions);

    Device = PhysicalDevice.createDevice(device_info);
    Queue = Device.getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::array<vk::DescriptorPoolSize, 1> pool_sizes = {vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1)};
    const vk::DescriptorPoolCreateInfo pool_info{vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, pool_sizes};
    DescriptorPool = Device.createDescriptorPool(pool_info);
}

void VulkanContext::Uninit() {
    Device.destroyDescriptorPool(DescriptorPool, nullptr);

#ifdef VKB_DEBUG
    auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)Instance.getProcAddr("vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(static_cast<VkInstance>(Instance), static_cast<VkDebugReportCallbackEXT>(g_DebugReport), nullptr);
#endif

    Device.destroy(nullptr);
    Instance.destroy(nullptr);
}

vk::PhysicalDevice VulkanContext::FindPhysicalDevice() const {
    const auto physicalDevices = Instance.enumeratePhysicalDevices();
    if (physicalDevices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physicalDevices) {
        vk::PhysicalDeviceProperties properties = device.getProperties();
        if (properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }

    return !physicalDevices.empty() ? physicalDevices[0] : vk::PhysicalDevice{};
}
