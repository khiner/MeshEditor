#pragma once

#include "numeric/vec2.h"

#include <vulkan/vulkan.hpp>

#include <functional>

namespace mvk {
struct ImageResource {
    vk::UniqueDeviceMemory Memory;
    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::Extent3D Extent;
};

struct ImGuiTexture {
    ImGuiTexture(vk::Device, vk::ImageView, vec2 uv0 = {0, 0}, vec2 uv1 = {1, 1});
    ~ImGuiTexture();

    vk::UniqueSampler Sampler;
    const vk::DescriptorSet DescriptorSet;
    const vec2 Uv0, Uv1;
};

uint32_t FindMemoryType(vk::PhysicalDevice, uint32_t, vk::MemoryPropertyFlags);

ImageResource CreateImage(vk::Device, vk::PhysicalDevice, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
void RecordBufferToSampledImageUpload(
    vk::CommandBuffer cb, vk::Buffer src, vk::Image dst, uint32_t width, uint32_t height, vk::ImageSubresourceRange subresource_range
);
} // namespace mvk
