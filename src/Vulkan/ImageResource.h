#pragma once

#include <vulkan/vulkan.hpp>

struct VulkanContext;

struct ImageResource {
    ImageResource(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    vk::UniqueImage Image;
    vk::Extent3D Extent;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

namespace ImageFormat {
const auto Color = vk::Format::eB8G8R8A8Unorm;
const auto Float = vk::Format::eR32G32B32A32Sfloat;
const auto Depth = vk::Format::eD32Sfloat;
} // namespace ImageFormat
