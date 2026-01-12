#include "Image.h"

#include "imgui_impl_vulkan.h"

namespace mvk {
ImGuiTexture::ImGuiTexture(vk::Device device, vk::ImageView image_view, vec2 uv0, vec2 uv1)
    : Sampler(device.createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear})),
      DescriptorSet(ImGui_ImplVulkan_AddTexture(*Sampler, image_view, VkImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal))),
      Uv0{uv0}, Uv1{uv1} {}

ImGuiTexture::~ImGuiTexture() {
    ImGui_ImplVulkan_RemoveTexture(DescriptorSet);
}

void ImGuiTexture::Draw(vec2 size) const {
    ImGui::Image(ImTextureID((void *)DescriptorSet), {size.x, size.y}, {Uv0.x, Uv0.y}, {Uv1.x, Uv1.y});
}

uint32_t FindMemoryType(vk::PhysicalDevice pd, uint32_t type_filter, vk::MemoryPropertyFlags prop_flags) {
    auto mem_props = pd.getMemoryProperties();
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & prop_flags) == prop_flags) return i;
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

ImageResource CreateImage(vk::Device d, vk::PhysicalDevice pd, vk::ImageCreateInfo image_info, vk::ImageViewCreateInfo view_info, vk::MemoryPropertyFlags mem_flags) {
    auto image = d.createImageUnique(image_info);
    const auto mem_reqs = d.getImageMemoryRequirements(*image);
    auto memory = d.allocateMemoryUnique({mem_reqs.size, FindMemoryType(pd, mem_reqs.memoryTypeBits, mem_flags)});
    d.bindImageMemory(*image, *memory, 0);
    view_info.image = *image;
    return {std::move(memory), std::move(image), d.createImageViewUnique(view_info), image_info.extent};
}
} // namespace mvk
