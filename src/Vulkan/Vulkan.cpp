#include "Vulkan.h"

#include "imgui_impl_vulkan.h"

ImGuiTexture::ImGuiTexture(vk::Device device, vk::ImageView image_view, vec2 uv0, vec2 uv1)
    : Sampler(device.createSamplerUnique({{}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear})),
      DescriptorSet(ImGui_ImplVulkan_AddTexture(*Sampler, image_view, VkImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal))),
      Uv0{uv0}, Uv1{uv1} {}

ImGuiTexture::~ImGuiTexture() {
    ImGui_ImplVulkan_RemoveTexture(DescriptorSet);
}

void ImGuiTexture::Render(vec2 size) const {
    ImGui::Image(ImTextureID((void *)DescriptorSet), {size.x, size.y}, {Uv0.x, Uv0.y}, {Uv1.x, Uv1.y});
}
