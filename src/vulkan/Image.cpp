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
} // namespace mvk
