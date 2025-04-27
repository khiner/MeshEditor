#include "VulkanContext.h"
#include "VulkanBuffer.h"

#include <format>
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

uint FindMemoryType(const vk::PhysicalDevice &physical_device, uint type_filter, vk::MemoryPropertyFlags prop_flags) {
    auto mem_props = physical_device.getMemoryProperties();
    for (uint i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
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
    BufferAllocator = std::make_unique<VulkanBufferAllocator>(PhysicalDevice, *Device, *Instance);
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

    CommandPool = Device->createCommandPoolUnique({vk::CommandPoolCreateFlagBits::eResetCommandBuffer, QueueFamily});
    TransferCommandBuffer = std::move(Device->allocateCommandBuffersUnique({*CommandPool, vk::CommandBufferLevel::ePrimary, 1}).front());
    RenderFence = Device->createFenceUnique({});
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

    // Note: `buffer.Size` is the _used_ size, not the allocated size.
    const auto required_bytes = offset + bytes;
    if (required_bytes > buffer.GetAllocatedSize()) {
        // Create a new buffer with the first large enough power of two.
        // Copy the old buffer into the new buffer (host and device), and replace the old buffer.
        const auto new_bytes = NextPowerOfTwo(required_bytes);
        auto new_buffer = BufferAllocator->CreateBuffer(buffer.Usage, new_bytes);
        // Host copy:
        new_buffer.HostBuffer.WriteRegion(buffer.HostBuffer.GetData(), 0, buffer.Size);
        // Host->device copy:
        const auto &cb = *TransferCommandBuffer;
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
    buffer.HostBuffer.WriteRegion(data, offset, bytes);

    // Copy data from the staging buffer to the device buffer.
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb.copyBuffer(*buffer.HostBuffer, *buffer.DeviceBuffer, vk::BufferCopy{offset, offset, bytes});
    cb.end();
    SubmitTransfer();
}

void VulkanContext::EraseBufferRegion(VulkanBuffer &buffer, vk::DeviceSize offset, vk::DeviceSize bytes) const {
    if (bytes == 0 || offset + bytes > buffer.Size) return;

    if (const auto move_bytes = buffer.Size - (offset + bytes); move_bytes > 0) {
        buffer.HostBuffer.MoveRegion(offset + bytes, offset, move_bytes);

        const auto &cb = *TransferCommandBuffer;
        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(*buffer.HostBuffer, *buffer.DeviceBuffer, vk::BufferCopy{offset, offset, move_bytes});
        cb.end();
        SubmitTransfer();
    }
    buffer.Size -= bytes;
}

void VulkanContext::WaitForRender() const {
    if (auto wait_result = Device->waitForFences(*RenderFence, VK_TRUE, UINT64_MAX); wait_result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to wait for fence: {}", vk::to_string(wait_result)));
    }
    Device->resetFences(*RenderFence);
}

// TODO Use separate fence/semaphores for buffer updates and rendering?
void VulkanContext::SubmitTransfer() const {
    vk::SubmitInfo submit;
    submit.setCommandBuffers(*TransferCommandBuffer);
    Queue.submit(submit, *RenderFence);
    WaitForRender();
}

ImageResource VulkanContext::CreateImage(vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_flags) const {
    auto image = Device->createImageUnique(image_info);
    const auto mem_reqs = Device->getImageMemoryRequirements(*image);
    auto memory = Device->allocateMemoryUnique({mem_reqs.size, FindMemoryType(PhysicalDevice, mem_reqs.memoryTypeBits, mem_flags)});
    Device->bindImageMemory(*image, *memory, 0);
    view_info.image = *image;
    return {std::move(memory), std::move(image), Device->createImageViewUnique(view_info), image_info.extent};
}

ImageResource VulkanContext::RenderBitmapToImage(const void *data, uint32_t width, uint32_t height) const {
    auto image = CreateImage(
        {{}, vk::ImageType::e2D, ImageFormat::Color, {width, height, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    // Write the bitmap into a staging buffer.
    const auto buffer_size = width * height * 4; // 4 bytes per pixel
    auto staging_buffer = BufferAllocator->CreateVmaBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc, MemoryUsage::CpuOnly);
    staging_buffer.WriteRegion(data, 0, buffer_size);

    // Record commands to copy from staging buffer to Vulkan image.
    const auto &cb = *TransferCommandBuffer;
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    // Transition the image layout to be ready for data transfer.
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            {}, // srcAccessMask
            vk::AccessFlagBits::eTransferWrite, // dstAccessMask
            vk::ImageLayout::eUndefined, // oldLayout
            vk::ImageLayout::eTransferDstOptimal, // newLayout
            VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
            VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
            *image.Image, // image
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresourceRange
        }
    );

    // Copy buffer to image.
    cb.copyBufferToImage(
        *staging_buffer, *image.Image, vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{
            0, // bufferOffset
            0, // bufferRowLength (tightly packed)
            0, // bufferImageHeight
            {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, // imageSubresource
            {0, 0, 0}, // imageOffset
            {width, height, 1} // imageExtent
        }
    );

    // Transition the image layout to be ready for shader sampling.
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{
            vk::AccessFlagBits::eTransferWrite, // srcAccessMask
            vk::AccessFlagBits::eShaderRead, // dstAccessMask
            vk::ImageLayout::eTransferDstOptimal, // oldLayout
            vk::ImageLayout::eShaderReadOnlyOptimal, // newLayout
            VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
            VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
            *image.Image, // image
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresourceRange
        }
    );

    cb.end();
    SubmitTransfer();

    return image;
}

#include "imgui_impl_vulkan.h"

ImGuiTexture::ImGuiTexture(vk::Device device, vk::ImageView image_view, vec2 uv0, vec2 uv1)
    : Sampler(device.createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear})),
      DescriptorSet(ImGui_ImplVulkan_AddTexture(*Sampler, image_view, VkImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal))),
      Uv0{uv0}, Uv1{uv1} {}

ImGuiTexture::~ImGuiTexture() {
    ImGui_ImplVulkan_RemoveTexture(DescriptorSet);
}

void ImGuiTexture::Render(vec2 size) const {
    ImGui::Image(ImTextureID((void *)DescriptorSet), {size.x, size.y}, {Uv0.x, Uv0.y}, {Uv1.x, Uv1.y});
}
