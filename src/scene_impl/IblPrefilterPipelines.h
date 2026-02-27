#pragma once

#include "Shader.h"
#include <vulkan/vulkan.hpp>

#include <array>

struct IblPrefilterPipelines {
    // Descriptor set layout: binding 0 = COMBINED_IMAGE_SAMPLER, binding 1 = STORAGE_IMAGE.
    // Push constants: 12 bytes max (SpecularPrefilter uses FaceSize+SourceSize+Roughness; others use fewer).
    // Used only during IBL prefiltering; not related to the global bindless system.
    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline EquirectToCubemap;
    vk::UniquePipeline DiffuseIrradiance;
    vk::UniquePipeline SpecularPrefilter;

    IblPrefilterPipelines(vk::Device device) {
        const std::array bindings{
            vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
            vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        };
        DescriptorSetLayout = device.createDescriptorSetLayoutUnique({{}, bindings});
        const vk::PushConstantRange push_range{vk::ShaderStageFlagBits::eCompute, 0, 12};
        PipelineLayout = device.createPipelineLayoutUnique({{}, *DescriptorSetLayout, push_range});
        EquirectToCubemap = CreatePipeline(device, "EquirectToCubemap.comp");
        DiffuseIrradiance = CreatePipeline(device, "DiffuseIrradiance.comp");
        SpecularPrefilter = CreatePipeline(device, "SpecularPrefilter.comp");
    }

private:
    vk::UniquePipeline CreatePipeline(vk::Device device, const char *shader_name) {
        Shaders shaders{{{ShaderType::eCompute, shader_name}}};
        const auto stages = shaders.CompileAll(device);
        return device.createComputePipelineUnique({}, {{}, stages.front(), *PipelineLayout}).value;
    }
};
