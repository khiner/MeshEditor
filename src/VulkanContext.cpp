#include "VulkanContext.h"

void VulkanContext::Init(std::vector<const char *> extensions) {
    // Create instance.
    {
        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        // Enumerate available extensions.
        uint32_t properties_count;
        vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, nullptr);
        std::vector<VkExtensionProperties> properties(properties_count);
        CheckVk(vkEnumerateInstanceExtensionProperties(nullptr, &properties_count, properties.data()));

        // Enable required extensions.
        if (IsExtensionAvailable(properties, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        // Enable validation layers.
#ifdef VKB_DEBUG
        const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
        create_info.enabledLayerCount = 1;
        create_info.ppEnabledLayerNames = layers;
        extensions.push_back("VK_EXT_debug_report");
#endif

        // Create Vulkan instance.
        create_info.enabledExtensionCount = (uint32_t)extensions.size();
        create_info.ppEnabledExtensionNames = extensions.data();
        CheckVk(vkCreateInstance(&create_info, nullptr, &Instance));

        // Setup the debug report callback.
#ifdef VKB_DEBUG
        auto vkCreateDebugReportCallbackEXT = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(nullptr, "vkCreateDebugReportCallbackEXT");
        IM_ASSERT(vkCreateDebugReportCallbackEXT != nullptr);
        VkDebugReportCallbackCreateInfoEXT debug_report_ci = {};
        debug_report_ci.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
        debug_report_ci.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT | VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT;
        debug_report_ci.pfnCallback = debug_report;
        debug_report_ci.pUserData = nullptr;
        err = vkCreateDebugReportCallbackEXT(nullptr, &debug_report_ci, nullptr, &g_DebugReport);
        check_vk_result(err);
#endif
    }

    // Select physical device (GPU).
    PhysicalDevice = FindPhysicalDevice();

    // Select graphics queue family.
    {
        uint32_t count;
        vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &count, nullptr);
        VkQueueFamilyProperties *queues = (VkQueueFamilyProperties *)malloc(sizeof(VkQueueFamilyProperties) * count);
        vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &count, queues);
        for (uint32_t i = 0; i < count; i++)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                QueueFamily = i;
                break;
            }
        free(queues);
        if (QueueFamily == (uint32_t)-1) throw std::runtime_error("No graphics queue family found.");
    }

    // Create logical device (with 1 queue).
    {
        std::vector<const char *> device_extensions;
        device_extensions.push_back("VK_KHR_swapchain");

        // Enumerate physical device extension properties.
        uint32_t properties_count;
        std::vector<VkExtensionProperties> properties;
        vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &properties_count, nullptr);
        properties.resize(properties_count);
        vkEnumerateDeviceExtensionProperties(PhysicalDevice, nullptr, &properties_count, properties.data());
#ifdef VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
        if (IsExtensionAvailable(properties, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME))
            device_extensions.push_back(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
#endif

        const float queue_priority[] = {1.0f};
        VkDeviceQueueCreateInfo queue_info[1] = {};
        queue_info[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info[0].queueFamilyIndex = QueueFamily;
        queue_info[0].queueCount = 1;
        queue_info[0].pQueuePriorities = queue_priority;
        VkDeviceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        create_info.queueCreateInfoCount = sizeof(queue_info) / sizeof(queue_info[0]);
        create_info.pQueueCreateInfos = queue_info;
        create_info.enabledExtensionCount = (uint32_t)device_extensions.size();
        create_info.ppEnabledExtensionNames = device_extensions.data();
        CheckVk(vkCreateDevice(PhysicalDevice, &create_info, nullptr, &Device));
        vkGetDeviceQueue(Device, QueueFamily, 0, &Queue);
    }

    // Create descriptor pool.
    // We only require a single combined image sampler descriptor for the font image and associated descriptor set.
    // Loading e.g. additional textures will require additional pool sizes.
    {
        VkDescriptorPoolSize pool_sizes[] = {
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        };
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = pool_sizes;
        CheckVk(vkCreateDescriptorPool(Device, &pool_info, nullptr, &DescriptorPool));
    }
}

void VulkanContext::Uninit() {
    vkDestroyDescriptorPool(Device, DescriptorPool, nullptr);

#ifdef VKB_DEBUG
    auto vkDestroyDebugReportCallbackEXT = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(nullptr, "vkDestroyDebugReportCallbackEXT");
    vkDestroyDebugReportCallbackEXT(nullptr, g_DebugReport, nullptr);
#endif

    vkDestroyDevice(Device, nullptr);
    vkDestroyInstance(Instance, nullptr);
}

VkPhysicalDevice VulkanContext::FindPhysicalDevice() const {
    uint32_t gpu_count;
    CheckVk(vkEnumeratePhysicalDevices(Instance, &gpu_count, nullptr));
    if (gpu_count <= 0) throw std::runtime_error("No Vulkan devices found.");

    std::vector<VkPhysicalDevice> gpus(gpu_count);
    CheckVk(vkEnumeratePhysicalDevices(Instance, &gpu_count, gpus.data()));

    for (VkPhysicalDevice &device : gpus) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return device;
    }

    return gpu_count > 0 ? gpus[0] : VK_NULL_HANDLE;
}
