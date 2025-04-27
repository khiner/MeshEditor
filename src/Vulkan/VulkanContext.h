#pragma once

#include "VulkanBuffer.h"
#include "numeric/vec2.h"

#include <vector>

using uint = u_int32_t;

struct ImageResource {
    vk::UniqueDeviceMemory Memory;
    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::Extent3D Extent;
};

struct ImGuiTexture {
    ImGuiTexture(vk::Device, vk::ImageView, vec2 uv0 = {0, 0}, vec2 uv1 = {1, 1});
    ~ImGuiTexture();

    void Render(vec2 size) const;

private:
    vk::UniqueSampler Sampler;
    vk::DescriptorSet DescriptorSet;
    const vec2 Uv0, Uv1; // UV coordinates.
};

namespace ImageFormat {
const auto Color = vk::Format::eB8G8R8A8Unorm;
const auto Float = vk::Format::eR32G32B32A32Sfloat;
const auto Depth = vk::Format::eD32Sfloat;
} // namespace ImageFormat

struct VulkanBufferAllocator;

struct VulkanContext {
    VulkanContext(std::vector<const char *> extensions);
    ~VulkanContext() = default; // Using unique handles, so no need to manually destroy anything.

    vk::UniqueInstance Instance;
    vk::PhysicalDevice PhysicalDevice;
    vk::UniqueDevice Device;
    std::unique_ptr<VulkanBufferAllocator> BufferAllocator;

    uint QueueFamily{uint(-1)};
    vk::Queue Queue;
    vk::UniquePipelineCache PipelineCache;
    vk::UniqueDescriptorPool DescriptorPool;

    vk::UniqueCommandPool CommandPool;
    vk::UniqueCommandBuffer TransferCommandBuffer;
    vk::UniqueFence RenderFence;

    // Uses `buffer.Size` if `bytes` is not set.
    // Automatically grows the buffer if the buffer is too small (using the nearest large enough power of 2).
    void UpdateBuffer(VulkanBuffer &, const void *data, vk::DeviceSize offset = 0, vk::DeviceSize bytes = 0) const;
    template<typename T> void UpdateBuffer(VulkanBuffer &buffer, const std::vector<T> &data) const {
        UpdateBuffer(buffer, data.data(), 0, sizeof(T) * data.size());
    }

    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // This is for dynamic buffers, and it doesn't free memory, so the allocated size will be greater than the used size.
    void EraseBufferRegion(VulkanBuffer &, vk::DeviceSize offset, vk::DeviceSize bytes) const;

    VulkanBuffer CreateBuffer(vk::BufferUsageFlags flags, const void *data, vk::DeviceSize bytes) const {
        auto buffer = BufferAllocator->CreateBuffer(flags, bytes);
        buffer.HostBuffer.WriteRegion(data, 0, bytes);
        // Copy data from the staging buffer to the device buffer.
        const auto &cb = *TransferCommandBuffer;
        cb.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        cb.copyBuffer(*buffer.HostBuffer, *buffer.DeviceBuffer, vk::BufferCopy{0, 0, bytes});
        cb.end();
        SubmitTransfer();
        return buffer;
    }
    void WaitForRender() const;
    void SubmitTransfer() const;

    ImageResource CreateImage(vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal) const;
    ImageResource RenderBitmapToImage(const void *data, uint32_t width, uint32_t height) const;
};
