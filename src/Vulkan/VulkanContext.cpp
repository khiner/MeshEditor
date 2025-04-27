#include "VulkanContext.h"

#include <iostream>
#include <numeric>
#include <ranges>

using std::ranges::any_of, std::ranges::distance, std::ranges::find_if;

/*
After upgrading from Vulkan SDK 1.3.290.0 to 1.4.304.0, I get:
Undefined symbols for architecture arm64:
"_vkCreateDebugUtilsMessengerEXT", referenced from:
    vk::detail::DispatchLoaderStatic::vkCreateDebugUtilsMessengerEXT(VkInstance_T*, VkDebugUtilsMessengerCreateInfoEXT const*, VkAllocationCallbacks const*, VkDebugUtilsMessengerEXT_T**) const in VulkanContext.cpp.o
ld: symbol(s) not found for architecture arm64
So disabling the debug messenger for now.
namespace {
vk::Bool32 DebugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data
) {
    (void)user_data;
    const char *severity_str = "";
    switch (severity) {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose: severity_str = "Verbose"; break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo: severity_str = "Info"; break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning: severity_str = "Warning"; break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError: severity_str = "Error"; break;
        default: break;
    }

    std::string type_str;
    if (type & vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral) {
        type_str = "General";
    }
    if (type & vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation) {
        type_str = "Validation";
    }
    if (type & vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance) {
        type_str = "Performance";
    }

    std::cerr << "[Vulkan|" << severity_str << "|" << type_str << "]" << ": " << callback_data->pMessage << '\n';
    return vk::False;
}
} // namespace
*/

namespace {
// Find a discrete GPU, or the first available (integrated) GPU.
vk::PhysicalDevice FindPhysicalDevice(const vk::UniqueInstance &instance) {
    const auto physical_devices = instance->enumeratePhysicalDevices();
    if (physical_devices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physical_devices) {
        if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }
    return physical_devices[0];
}
} // namespace

VulkanContext::VulkanContext(std::vector<const char *> enabled_extensions) {
    const auto IsLayerAvailable = [&](std::string_view layer) {
        static const auto available_layers = vk::enumerateInstanceLayerProperties();
        return any_of(available_layers, [layer](const auto &prop) { return strcmp(prop.layerName, layer.data()) == 0; });
    };
    std::vector<const char *> enabled_layers;
    const auto AddLayerIfAvailable = [&](std::string_view layer) {
        if (IsLayerAvailable(layer)) {
            enabled_layers.push_back(layer.data());
        } else {
            std::cerr << "Warning: Validation layer " << layer << " not available." << std::endl;
        }
    };
    AddLayerIfAvailable("VK_LAYER_KHRONOS_validation");

    const auto instance_props = vk::enumerateInstanceExtensionProperties();
    vk::InstanceCreateFlags flags;
    const auto IsExtensionAvailable = [&](std::string_view extension) {
        return any_of(instance_props, [extension](const auto &prop) { return strcmp(prop.extensionName, extension.data()) == 0; });
    };
    const auto AddExtensionIfAvailable = [&](std::string_view extension, vk::InstanceCreateFlags flag = {}) {
        if (IsExtensionAvailable(extension)) {
            enabled_extensions.push_back(extension.data());
            flags |= flag;
            return true;
        } else {
            std::cerr << "Warning: Extension " << extension << " not available." << std::endl;
            return false;
        }
    };
    AddExtensionIfAvailable(vk::KHRPortabilityEnumerationExtensionName, vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
    AddExtensionIfAvailable(vk::EXTDebugUtilsExtensionName);

    const vk::ApplicationInfo app{"", {}, "", {}, VK_API_VERSION_1_3};
    Instance = vk::createInstanceUnique({flags, &app, enabled_layers, enabled_extensions});
    /*
    if (AddExtensionIfAvailable(vk::EXTDebugUtilsExtensionName)) {
        const auto _ = Instance->createDebugUtilsMessengerEXT(
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
                nullptr
            }
        );
    }
    */

    PhysicalDevice = FindPhysicalDevice(Instance);

    const auto qfp = PhysicalDevice.getQueueFamilyProperties();
    const auto qfp_find_graphics_it = find_if(qfp, [](const auto &qfp) { return bool(qfp.queueFlags & vk::QueueFlagBits::eGraphics); });
    if (qfp_find_graphics_it == qfp.end()) throw std::runtime_error("No graphics queue family found.");

    QueueFamily = distance(qfp.begin(), qfp_find_graphics_it);
    if (!PhysicalDevice.getFeatures().fillModeNonSolid) {
        throw std::runtime_error("`fillModeNonSolid` is not supported, but is needed for line rendering.");
    }

    vk::PhysicalDeviceFeatures device_features{};
    device_features.fillModeNonSolid = VK_TRUE;

    // Create logical device (with one queue).
    static const std::vector<const char *> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME, "VK_KHR_portability_subset"};
    static constexpr std::array queue_priority{1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions, &device_features});
    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::vector pool_sizes{
        // Image samplers:
        // 1) (2D filled shape) silhouette of the active mesh
        // 2) (2D line) silhouette of the active mesh, after edge detection
        // 3) Final scene texture sampler
        // 4) ImGui fonts
        // 5) SVG image texture
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 5},
        // All uniform buffer descriptors used across all shaders.
        {vk::DescriptorType::eUniformBuffer, 7},
    };
    const uint max_sets = std::accumulate(
        pool_sizes.begin(), pool_sizes.end(), 0u,
        [](uint sum, const vk::DescriptorPoolSize &pool_size) { return sum + pool_size.descriptorCount; }
    );
    DescriptorPool = Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, max_sets, pool_sizes});
}
