#pragma once

#include "numeric/vec2.h"

#include <span>
#include <vulkan/vulkan.hpp>

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

ImageResource CreateImage(vk::Device, vk::PhysicalDevice, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags = vk::MemoryPropertyFlagBits::eDeviceLocal);
// Upload staging bytes to `dst` via `copies`, leaving it eShaderReadOnlyOptimal for `dst_stage`.
void RecordBufferToImageUpload(
    vk::CommandBuffer, vk::Buffer src, vk::Image dst, std::span<const vk::BufferImageCopy>, vk::ImageSubresourceRange,
    vk::PipelineStageFlags dst_stage = vk::PipelineStageFlagBits::eFragmentShader
);
// Single full-image mip-0 upload.
void RecordBufferToSampledImageUpload(
    vk::CommandBuffer, vk::Buffer src, vk::Image dst, uint32_t width, uint32_t height, vk::ImageSubresourceRange subresource_range,
    vk::DeviceSize buffer_offset = 0
);
// Record a single image-layout transition (one pipeline barrier).
void TransitionImage(
    vk::CommandBuffer, vk::PipelineStageFlags src_stage, vk::PipelineStageFlags dst_stage,
    vk::AccessFlags src_access, vk::AccessFlags dst_access, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
    vk::Image, vk::ImageSubresourceRange
);
// Copy a color sub-rect (mip 0) of `src` to `dst`, transitioning eShaderReadOnlyOptimal -> transfer-src and back.
void RecordImageToBufferCopy(vk::CommandBuffer, vk::Image src, vk::Buffer dst, vk::Offset3D, vk::Extent2D);
} // namespace mvk
