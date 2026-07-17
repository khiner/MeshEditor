#pragma once

#include "numeric/vec2.h"

#include <bit>
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
// Single-sampled optimal-tiling 2D image with a full-range view. Aspect derives from the format.
ImageResource CreateImage2D(vk::Device, vk::PhysicalDevice, vk::Format, vk::Extent2D, vk::ImageUsageFlags, uint32_t mip_levels = 1);
// Cubemap (6 layers) with a full-range eCube view.
ImageResource CreateImageCube(vk::Device, vk::PhysicalDevice, vk::Format, uint32_t size, vk::ImageUsageFlags, uint32_t mip_levels = 1);
vk::UniqueFramebuffer CreateFramebuffer(vk::Device, vk::RenderPass, std::initializer_list<vk::ImageView> views, vk::Extent2D);

// Mip count for a full chain down to 1x1.
constexpr uint32_t MipLevelCount(uint32_t width, uint32_t height) {
    const auto max_dim = std::max(width, height);
    return max_dim > 0 ? uint32_t(std::bit_width(max_dim)) : 1u;
}

// Sampler with identical min/mag filtering and one address mode on all axes.
constexpr vk::SamplerCreateInfo SamplerInfo(vk::Filter filter, vk::SamplerMipmapMode mipmap_mode, vk::SamplerAddressMode address_mode, float max_lod = 0.f) {
    return {{}, filter, filter, mipmap_mode, address_mode, address_mode, address_mode, 0.f, VK_FALSE, 1.f, VK_FALSE, vk::CompareOp::eNever, 0.f, max_lod, vk::BorderColor::eIntOpaqueBlack, VK_FALSE};
}
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
// Blit-generate mips 1..N-1 from mip 0. Expects every mip in eTransferDstOptimal, and leaves all
// mips eShaderReadOnlyOptimal for `dst_stage`.
void GenerateMipChain(
    vk::CommandBuffer, vk::Image, uint32_t width, uint32_t height, uint32_t mip_levels, uint32_t layers = 1,
    vk::PipelineStageFlags dst_stage = vk::PipelineStageFlagBits::eFragmentShader
);
} // namespace mvk
