#include "Image.h"
#include "FindMemoryType.h"

namespace mvk {
ImageResource CreateImage(vk::Device d, vk::PhysicalDevice pd, vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_flags) {
    auto image = d.createImageUnique(image_info);
    const auto mem_reqs = d.getImageMemoryRequirements(*image);
    auto memory = d.allocateMemoryUnique({mem_reqs.size, FindMemoryType(pd, mem_reqs.memoryTypeBits, mem_flags)});
    d.bindImageMemory(*image, *memory, 0);
    view_info.image = *image;
    return {std::move(memory), std::move(image), d.createImageViewUnique(view_info), image_info.extent};
}

ImageResource CreateImage2D(vk::Device d, vk::PhysicalDevice pd, vk::Format format, vk::Extent2D extent, vk::ImageUsageFlags usage, uint32_t mip_levels) {
    const auto aspect = format == vk::Format::eD32Sfloat ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
    return CreateImage(
        d, pd,
        {{}, vk::ImageType::e2D, format, vk::Extent3D{extent, 1}, mip_levels, 1, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::e2D, format, {}, {aspect, 0, mip_levels, 0, 1}}
    );
}

ImageResource CreateImageCube(vk::Device d, vk::PhysicalDevice pd, vk::Format format, uint32_t size, vk::ImageUsageFlags usage, uint32_t mip_levels) {
    return CreateImage(
        d, pd,
        {vk::ImageCreateFlagBits::eCubeCompatible, vk::ImageType::e2D, format, {size, size, 1}, mip_levels, 6, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usage, vk::SharingMode::eExclusive},
        {{}, {}, vk::ImageViewType::eCube, format, {}, {vk::ImageAspectFlagBits::eColor, 0, mip_levels, 0, 6}}
    );
}

vk::UniqueFramebuffer CreateFramebuffer(vk::Device d, vk::RenderPass render_pass, std::initializer_list<vk::ImageView> views, vk::Extent2D extent) {
    return d.createFramebufferUnique({{}, render_pass, uint32_t(views.size()), views.begin(), extent.width, extent.height, 1});
}

void TransitionImage(
    vk::CommandBuffer cb, vk::PipelineStageFlags src_stage, vk::PipelineStageFlags dst_stage,
    vk::AccessFlags src_access, vk::AccessFlags dst_access, vk::ImageLayout old_layout, vk::ImageLayout new_layout, vk::Image image, vk::ImageSubresourceRange range
) {
    cb.pipelineBarrier(src_stage, dst_stage, {}, {}, {}, vk::ImageMemoryBarrier{src_access, dst_access, old_layout, new_layout, {}, {}, image, range});
}

void RecordBufferToImageUpload(
    vk::CommandBuffer cb, vk::Buffer src, vk::Image dst, std::span<const vk::BufferImageCopy> copies,
    vk::ImageSubresourceRange range, vk::PipelineStageFlags dst_stage
) {
    TransitionImage(
        cb, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
        {}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, dst, range
    );
    cb.copyBufferToImage(src, dst, vk::ImageLayout::eTransferDstOptimal, {uint32_t(copies.size()), copies.data()});
    TransitionImage(
        cb, vk::PipelineStageFlagBits::eTransfer, dst_stage,
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, dst, range
    );
}

void RecordBufferToSampledImageUpload(
    vk::CommandBuffer cb, vk::Buffer src, vk::Image dst, uint32_t width, uint32_t height,
    vk::ImageSubresourceRange subresource_range, vk::DeviceSize buffer_offset
) {
    const vk::BufferImageCopy copy{buffer_offset, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, height, 1}};
    RecordBufferToImageUpload(cb, src, dst, {&copy, 1}, subresource_range);
}

void GenerateMipChain(vk::CommandBuffer cb, vk::Image image, uint32_t width, uint32_t height, uint32_t mip_levels, uint32_t layers, vk::PipelineStageFlags dst_stage) {
    int32_t mip_w = width, mip_h = height;
    for (uint32_t mip = 1; mip < mip_levels; ++mip) {
        const int32_t next_w = std::max(1, mip_w / 2), next_h = std::max(1, mip_h / 2);
        TransitionImage(
            cb, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eTransferRead,
            vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eTransferSrcOptimal, image, {vk::ImageAspectFlagBits::eColor, mip - 1, 1, 0, layers}
        );
        cb.blitImage(
            image, vk::ImageLayout::eTransferSrcOptimal, image, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageBlit{
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip - 1, 0, layers},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{mip_w, mip_h, 1}},
                vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, mip, 0, layers},
                {vk::Offset3D{0, 0, 0}, vk::Offset3D{next_w, next_h, 1}},
            },
            vk::Filter::eLinear
        );
        TransitionImage(
            cb, vk::PipelineStageFlagBits::eTransfer, dst_stage, vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, image, {vk::ImageAspectFlagBits::eColor, mip - 1, 1, 0, layers}
        );
        mip_w = next_w;
        mip_h = next_h;
    }
    TransitionImage(
        cb, vk::PipelineStageFlagBits::eTransfer, dst_stage, vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, image, {vk::ImageAspectFlagBits::eColor, mip_levels - 1, 1, 0, layers}
    );
}

void RecordImageToBufferCopy(vk::CommandBuffer cb, vk::Image src, vk::Buffer dst, vk::Offset3D offset, vk::Extent2D extent) {
    const vk::ImageSubresourceRange color_mip0{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    TransitionImage(
        cb, vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer,
        vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, src, color_mip0
    );
    cb.copyImageToBuffer(
        src, vk::ImageLayout::eTransferSrcOptimal, dst,
        vk::BufferImageCopy{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, offset, {extent.width, extent.height, 1}}
    );
    TransitionImage(
        cb, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, src, color_mip0
    );
}
} // namespace mvk
