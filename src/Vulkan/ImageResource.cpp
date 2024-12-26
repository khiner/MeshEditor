#include "ImageResource.h"

#include "VulkanContext.h"

ImageResource::ImageResource(const VulkanContext &vc, vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_props)
    : Extent(image_info.extent), Image(vc.Device->createImageUnique(image_info)) {
    const auto mem_reqs = vc.Device->getImageMemoryRequirements(*Image);
    Memory = vc.Device->allocateMemoryUnique({mem_reqs.size, vc.FindMemoryType(mem_reqs.memoryTypeBits, mem_props)});
    vc.Device->bindImageMemory(*Image, *Memory, 0);
    view_info.image = *Image;
    View = vc.Device->createImageViewUnique(view_info);
}
