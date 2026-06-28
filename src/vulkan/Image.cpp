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
