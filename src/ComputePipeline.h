#pragma once

#include "Shader.h"
#include "Vulkan/UniqueBuffers.h"

#include <vulkan/vulkan.hpp>

struct ComputePipeline {
    ComputePipeline(vk::Device device, vk::DescriptorPool descriptor_pool, Shaders &&shaders);
    ComputePipeline(ComputePipeline &&) = default;
    ~ComputePipeline() = default;

    void Bind(vk::CommandBuffer cmd) const;
    void Dispatch(vk::CommandBuffer cmd, uint32_t x, uint32_t y = 1, uint32_t z = 1) const;

    // Update descriptor set with buffer
    void UpdateDescriptorSet(uint32_t binding, vk::Buffer buffer, vk::DescriptorType type) const;

    // Update descriptor set with image/sampler
    void UpdateDescriptorSet(uint32_t binding, vk::ImageView image_view, vk::Sampler sampler) const;

    vk::Device Device;
    Shaders Shaders;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;
};
