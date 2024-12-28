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

    const auto _ = Instance->createDebugUtilsMessengerEXTUnique(
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
    static constexpr std::array queue_priority{1.0f};
    const vk::DeviceQueueCreateInfo queue_info{{}, QueueFamily, 1, queue_priority.data()};
    Device = PhysicalDevice.createDeviceUnique({{}, queue_info, {}, device_extensions, &device_features});

    BufferAllocator = std::make_unique<VulkanBufferAllocator>(PhysicalDevice, *Device, *Instance);

    Queue = Device->getQueue(QueueFamily, 0);

    // Create descriptor pool.
    const std::vector pool_sizes{
        // Image samplers:
        // 1) (2D filled shape) silhouette of the selected mesh
        // 2) (2D line) silhouette of the selected mesh, after edge detection
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

    const auto &cb = *TransferCommandBuffers.front();

    // Note: `buffer.Size` is the _used_ size, not the allocated size.
    const auto required_bytes = offset + bytes;
    if (required_bytes > buffer.GetAllocatedSize()) {
        // Create a new buffer with the first large enough power of two.
        // Copy the old buffer into the new buffer (host and device), and replace the old buffer.
        const auto new_bytes = NextPowerOfTwo(required_bytes);
        VulkanBuffer new_buffer = CreateBuffer(buffer.Usage, new_bytes);
        // Host copy:
        char *host_data = buffer.HostBuffer.MapMemory();
        new_buffer.HostBuffer.WriteRegion(host_data, 0, buffer.Size);
        buffer.HostBuffer.UnmapMemory();
        // Host->device copy:
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
    cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    cb.copyBuffer(*buffer.HostBuffer, *buffer.DeviceBuffer, vk::BufferCopy{offset, offset, bytes});
    cb.end();

    SubmitTransfer();
}

void VulkanContext::EraseBufferRegion(VulkanBuffer &buffer, vk::DeviceSize offset, vk::DeviceSize bytes) const {
    if (bytes == 0 || offset + bytes > buffer.Size) return;

    if (const auto move_bytes = buffer.Size - (offset + bytes); move_bytes > 0) {
        buffer.HostBuffer.MoveRegion(offset + bytes, offset, move_bytes);

        const auto &cb = *TransferCommandBuffers.front();
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
    submit.setCommandBuffers(*TransferCommandBuffers.front());
    Queue.submit(submit, *RenderFence);
    WaitForRender();
}

std::unique_ptr<ImageResource> VulkanContext::CreateImage(vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_flags) const {
    auto image = Device->createImageUnique(image_info);
    const auto mem_reqs = Device->getImageMemoryRequirements(*image);
    auto memory = Device->allocateMemoryUnique({mem_reqs.size, FindMemoryType(mem_reqs.memoryTypeBits, mem_flags)});
    Device->bindImageMemory(*image, *memory, 0);
    view_info.image = *image;
    auto view = Device->createImageViewUnique(view_info);
    return std::make_unique<ImageResource>(std::move(memory), std::move(image), std::move(view));
}
std::unique_ptr<ImageResource> VulkanContext::RenderBitmapToImage(const void *data, uint32_t width, uint32_t height) const {
    auto image = CreateImage(
        {{}, vk::ImageType::e2D, ImageFormat::Color, {width, height, 1}, 1, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, ImageFormat::Color, {}, {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}}
    );
    // Write the bitmap into a staging buffer.
    const auto buffer_size = width * height * 4; // 4 bytes per pixel
    auto staging_buffer = BufferAllocator->CreateBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc, MemoryUsage::CpuOnly);
    staging_buffer.WriteRegion(data, 0, buffer_size);

    // Record commands to copy from staging buffer to Vulkan image.
    const auto &cb = *TransferCommandBuffers.front();
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
            *image->Image, // image
            {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} // subresourceRange
        }
    );

    // Copy buffer to image.
    cb.copyBufferToImage(
        *staging_buffer, *image->Image, vk::ImageLayout::eTransferDstOptimal,
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
            *image->Image, // image
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
