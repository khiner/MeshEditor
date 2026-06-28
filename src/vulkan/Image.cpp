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

void RecordBufferToSampledImageUpload(
    vk::CommandBuffer cb, vk::Buffer src, vk::Image dst, uint32_t width, uint32_t height,
    vk::ImageSubresourceRange subresource_range, vk::DeviceSize buffer_offset
) {
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer,
        {}, {}, {},
        vk::ImageMemoryBarrier{{}, vk::AccessFlagBits::eTransferWrite, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, {}, {}, dst, subresource_range}
    );

    cb.copyBufferToImage(
        src, dst, vk::ImageLayout::eTransferDstOptimal,
        vk::BufferImageCopy{buffer_offset, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, {0, 0, 0}, {width, height, 1}}
    );

    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader,
        {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, dst, subresource_range}
    );
}

void RecordImageToBufferCopy(vk::CommandBuffer cb, vk::Image src, vk::Buffer dst, vk::Offset3D offset, vk::Extent2D extent) {
    const vk::ImageSubresourceRange color_mip0{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eTransferSrcOptimal, {}, {}, src, color_mip0}
    );
    cb.copyImageToBuffer(
        src, vk::ImageLayout::eTransferSrcOptimal, dst,
        vk::BufferImageCopy{0, 0, 0, {vk::ImageAspectFlagBits::eColor, 0, 0, 1}, offset, {extent.width, extent.height, 1}}
    );
    cb.pipelineBarrier(
        vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader, {}, {}, {},
        vk::ImageMemoryBarrier{vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eShaderRead, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, {}, {}, src, color_mip0}
    );
}
} // namespace mvk
