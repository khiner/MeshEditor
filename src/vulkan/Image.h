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

    void Draw(vec2 size) const; // ImGui draw
    vk::DescriptorSet GetDescriptorSet() const { return DescriptorSet; }

private:
    vk::UniqueSampler Sampler;
    vk::DescriptorSet DescriptorSet;
    const vec2 Uv0, Uv1; // UV coordinates.
};

uint32_t FindMemoryType(vk::PhysicalDevice, uint32_t, vk::MemoryPropertyFlags);

ImageResource CreateImage(vk::Device, vk::PhysicalDevice, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
} // namespace mvk
