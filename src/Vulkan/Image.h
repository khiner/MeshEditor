#pragma once

#include "numeric/vec2.h"

#include <vulkan/vulkan.hpp>

#include <functional>
#include <span>

namespace mvk {
struct ImageResource {
    vk::UniqueDeviceMemory Memory;
    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::Extent3D Extent;
};

using BitmapToImage = std::function<ImageResource(std::span<const std::byte> data, uint32_t width, uint32_t height)>;

namespace ImageFormat {
constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
constexpr auto Float2 = vk::Format::eR32G32Sfloat;
constexpr auto Float4 = vk::Format::eR32G32B32A32Sfloat;
constexpr auto Depth = vk::Format::eD32Sfloat;
} // namespace ImageFormat

struct ImGuiTexture {
    ImGuiTexture(vk::Device, vk::ImageView, vec2 uv0 = {0, 0}, vec2 uv1 = {1, 1});
    ~ImGuiTexture();

    void Render(vec2 size) const;

private:
    vk::UniqueSampler Sampler;
    vk::DescriptorSet DescriptorSet;
    const vec2 Uv0, Uv1; // UV coordinates.
};

uint32_t FindMemoryType(vk::PhysicalDevice, uint32_t, vk::MemoryPropertyFlags);

ImageResource CreateImage(vk::Device, vk::PhysicalDevice, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
} // namespace mvk
