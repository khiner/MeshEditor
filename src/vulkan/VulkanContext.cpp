#include "VulkanContext.h"

#include <iostream>
#include <span>

vk::PhysicalDevice FindPhysicalDevice(const vk::UniqueInstance &instance) {
    const auto physical_devices = instance->enumeratePhysicalDevices();
    if (physical_devices.empty()) throw std::runtime_error("No Vulkan devices found.");

    for (const auto &device : physical_devices) {
        if (device.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) return device;
    }
    return physical_devices[0];
}

VulkanContext::VulkanContext(std::vector<const char *> enabled_extensions, bool with_swapchain) {
    const auto IsExtensionAvailable = [](std::span<const vk::ExtensionProperties> props, std::string_view extension) {
        return std::ranges::any_of(props, [extension](const auto &prop) { return std::strcmp(prop.extensionName, extension.data()) == 0; });
    };

    const auto IsLayerAvailable = [&](std::string_view layer) {
        static const auto available_layers = vk::enumerateInstanceLayerProperties();
        return std::ranges::any_of(available_layers, [layer](const auto &prop) { return std::strcmp(prop.layerName, layer.data()) == 0; });
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
    const auto AddExtensionIfAvailable = [&](std::string_view extension, vk::InstanceCreateFlags flag = {}) {
        if (IsExtensionAvailable(instance_props, extension)) {
            enabled_extensions.push_back(extension.data());
            flags |= flag;
            return true;
        }
        std::cerr << "Warning: Extension " << extension << " not available." << std::endl;
        return false;
    };
    AddExtensionIfAvailable(vk::KHRPortabilityEnumerationExtensionName, vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR);
    AddExtensionIfAvailable(vk::EXTDebugUtilsExtensionName);

    const vk::ApplicationInfo app{"", {}, "", {}, VkApiVersion};
    Instance = vk::createInstanceUnique({flags, &app, enabled_layers, enabled_extensions});
    PhysicalDevice = FindPhysicalDevice(Instance);

    const auto device_extensions_props = PhysicalDevice.enumerateDeviceExtensionProperties();
    const auto RequireDeviceExtension = [&](std::string_view extension) {
        if (!IsExtensionAvailable(device_extensions_props, extension)) {
            throw std::runtime_error(std::format("Required device extension {} is not available.", extension));
        }
    };
    const auto supported_features = PhysicalDevice.getFeatures2<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceDescriptorIndexingFeatures,
        vk::PhysicalDeviceBufferDeviceAddressFeatures,
        vk::PhysicalDeviceScalarBlockLayoutFeatures,
        vk::PhysicalDevice8BitStorageFeatures,
        vk::PhysicalDeviceShaderFloat16Int8Features>();
    const auto &supported_indexing = supported_features.get<vk::PhysicalDeviceDescriptorIndexingFeatures>();
    const auto &supported_bda = supported_features.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>();
    const auto &supported_scalar = supported_features.get<vk::PhysicalDeviceScalarBlockLayoutFeatures>();
    const auto &supported_8bit_storage = supported_features.get<vk::PhysicalDevice8BitStorageFeatures>();
    const auto &supported_int8 = supported_features.get<vk::PhysicalDeviceShaderFloat16Int8Features>();
    const auto RequireFeature = [](bool supported, std::string_view feature_name) {
        if (!supported) throw std::runtime_error(std::format("Required device feature {} is not available.", feature_name));
    };

    const auto qfp = PhysicalDevice.getQueueFamilyProperties();
    const auto qfp_find_graphics_it = std::ranges::find_if(qfp, [](const auto &q) { return bool(q.queueFlags & vk::QueueFlagBits::eGraphics); });
    if (qfp_find_graphics_it == qfp.end()) throw std::runtime_error("No graphics queue family found.");

    QueueFamily = std::distance(qfp.begin(), qfp_find_graphics_it);
    if (!PhysicalDevice.getFeatures().multiDrawIndirect) {
        throw std::runtime_error("`multiDrawIndirect` is not supported, but is needed for MDI submission.");
    }

    vk::PhysicalDeviceFeatures device_features{};
    device_features.multiDrawIndirect = VK_TRUE;
    device_features.fragmentStoresAndAtomics = VK_TRUE;
    device_features.independentBlend = VK_TRUE;

    std::vector<const char *> device_extensions;
    if (with_swapchain) {
        RequireDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    if (IsExtensionAvailable(device_extensions_props, "VK_KHR_portability_subset")) {
        device_extensions.push_back("VK_KHR_portability_subset");
    }
    RequireDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    RequireDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

    RequireFeature(supported_indexing.runtimeDescriptorArray, "runtimeDescriptorArray");
    RequireFeature(supported_indexing.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound");
    RequireFeature(supported_indexing.descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount");
    RequireFeature(supported_indexing.shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing");
    RequireFeature(supported_indexing.shaderStorageBufferArrayNonUniformIndexing, "shaderStorageBufferArrayNonUniformIndexing");
    RequireFeature(supported_indexing.descriptorBindingSampledImageUpdateAfterBind, "descriptorBindingSampledImageUpdateAfterBind");
    RequireFeature(supported_indexing.descriptorBindingStorageBufferUpdateAfterBind, "descriptorBindingStorageBufferUpdateAfterBind");
    RequireFeature(supported_indexing.descriptorBindingStorageImageUpdateAfterBind, "descriptorBindingStorageImageUpdateAfterBind");
    RequireFeature(supported_bda.bufferDeviceAddress, "bufferDeviceAddress");
    RequireFeature(supported_scalar.scalarBlockLayout, "scalarBlockLayout");
    RequireFeature(supported_8bit_storage.storageBuffer8BitAccess, "storageBuffer8BitAccess");
    RequireFeature(supported_8bit_storage.uniformAndStorageBuffer8BitAccess, "uniformAndStorageBuffer8BitAccess");
    RequireFeature(supported_int8.shaderInt8, "shaderInt8");

    vk::PhysicalDeviceBufferDeviceAddressFeatures buffer_device_address_features{};
    buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
    vk::PhysicalDeviceScalarBlockLayoutFeatures scalar_block_layout_features{};
    scalar_block_layout_features.scalarBlockLayout = VK_TRUE;
    vk::PhysicalDevice8BitStorageFeatures storage_8bit_features{};
    storage_8bit_features.storageBuffer8BitAccess = VK_TRUE;
    storage_8bit_features.uniformAndStorageBuffer8BitAccess = VK_TRUE;
    vk::PhysicalDeviceShaderFloat16Int8Features shader_int8_features{};
    shader_int8_features.shaderInt8 = VK_TRUE;
    shader_int8_features.pNext = &storage_8bit_features;
    storage_8bit_features.pNext = &scalar_block_layout_features;
    scalar_block_layout_features.pNext = &buffer_device_address_features;
    vk::PhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features{};
    descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    descriptor_indexing_features.shaderStorageImageArrayNonUniformIndexing = supported_indexing.shaderStorageImageArrayNonUniformIndexing;
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
    descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;
    descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    descriptor_indexing_features.descriptorBindingSampledImageUpdateAfterBind = supported_indexing.descriptorBindingSampledImageUpdateAfterBind;
    descriptor_indexing_features.descriptorBindingStorageBufferUpdateAfterBind = supported_indexing.descriptorBindingStorageBufferUpdateAfterBind;
    descriptor_indexing_features.descriptorBindingStorageImageUpdateAfterBind = supported_indexing.descriptorBindingStorageImageUpdateAfterBind;
    descriptor_indexing_features.pNext = &shader_int8_features;
    vk::PhysicalDeviceFeatures2 features2{};
    features2.features = device_features;
    features2.pNext = &descriptor_indexing_features;

    static constexpr std::array queue_priority{1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    vk::DeviceCreateInfo dci{{}, queue_info, {}, device_extensions, nullptr};
    dci.pNext = &features2;
    Device = PhysicalDevice.createDeviceUnique(dci);
    Queue = Device->getQueue(QueueFamily, 0);

    const std::array pool_sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 16},
        vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8},
        vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 4},
    };
    DescriptorPool = Device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 64, pool_sizes});
}
