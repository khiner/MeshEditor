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

    const uint FramebufferCount{1};
    vk::UniqueCommandPool CommandPool;
    std::vector<vk::UniqueCommandBuffer> CommandBuffers;
    std::vector<vk::UniqueCommandBuffer> TransferCommandBuffers;
    vk::UniqueFence RenderFence;

    // Find a discrete GPU, or the first available (integrated) GPU.
    vk::PhysicalDevice FindPhysicalDevice() const;
    uint FindMemoryType(uint type_filter, vk::MemoryPropertyFlags) const;

    // Create the staging and device buffers and their memory.
    // Assumes `buffer.Size` is set.
    VulkanBuffer CreateBuffer(vk::BufferUsageFlags, vk::DeviceSize) const;

    // Uses `buffer.Size` if `bytes` is not set.
    // Automatically grows the buffer if the buffer is too small (using the nearest large enough power of 2).
    void UpdateBuffer(VulkanBuffer &, const void *data, vk::DeviceSize offset = 0, vk::DeviceSize bytes = 0) const;

    // Erase a region of a buffer by moving the data after the region to the beginning of the region and reducing the buffer size.
    // This is for dynamic buffers, and it doesn't free memory, so the allocated size will be greater than the used size.
    void EraseBufferRegion(VulkanBuffer &, vk::DeviceSize offset, vk::DeviceSize bytes) const;

    template<typename T> VulkanBuffer CreateBuffer(vk::BufferUsageFlags flags, const std::vector<T> &data) const {
        const uint bytes = sizeof(T) * data.size();
        auto buffer = CreateBuffer(flags, bytes);
        UpdateBuffer(buffer, data.data(), 0, bytes);
        return buffer;
    }
    template<typename T, size_t N> VulkanBuffer CreateBuffer(vk::BufferUsageFlags flags, const std::array<T, N> &data) const {
        const uint bytes = sizeof(T) * N;
        auto buffer = CreateBuffer(flags, bytes);
        UpdateBuffer(buffer, data.data(), 0, bytes);
        return buffer;
    }
    template<typename T> void UpdateBuffer(VulkanBuffer &buffer, const std::vector<T> &data) const {
        UpdateBuffer(buffer, data.data(), 0, sizeof(T) * data.size());
    }

    void WaitForRender() const;
    void SubmitTransfer() const;

    std::unique_ptr<ImageResource> CreateImage(vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags mem_flags = vk::MemoryPropertyFlagBits::eDeviceLocal) const;
    std::unique_ptr<ImageResource> RenderBitmapToImage(const void *data, uint32_t width, uint32_t height) const;
};
