#pragma once

#include <filesystem>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.hpp>

namespace fs = std::filesystem;

namespace spirv_cross {
struct ShaderResources;
}

using ShaderType = vk::ShaderStageFlagBits;

struct Shaders {
    Shaders(std::unordered_map<ShaderType, fs::path> &&paths);
    Shaders(Shaders &&);
    ~Shaders();

    Shaders &operator=(Shaders &&);

    std::vector<vk::PipelineShaderStageCreateInfo> CompileAll(const vk::UniqueDevice &); // Populates `Modules`.
    std::vector<uint> Compile(ShaderType) const;

    std::unordered_map<ShaderType, fs::path> Paths; // Paths are relative to the `Shaders` directory.
    std::unordered_map<ShaderType, vk::UniqueShaderModule> Modules;
    std::unordered_map<ShaderType, std::unique_ptr<spirv_cross::ShaderResources>> Resources;
};

struct ShaderPipeline {
    ShaderPipeline(
        const vk::UniqueDevice &, Shaders &&,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
        bool test_depth = true, bool write_depth = true,
        vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1
    );
    virtual ~ShaderPipeline() = default;

    void Compile(const vk::UniqueRenderPass &); // Recompile all shaders and update `Pipeline`.

    const vk::UniqueDevice &Device;

    Shaders Shaders;
    vk::PipelineMultisampleStateCreateInfo MultisampleState;
    vk::PipelineColorBlendAttachmentState ColorBlendAttachment;
    vk::PipelineDepthStencilStateCreateInfo DepthStencilState;
    vk::PipelineVertexInputStateCreateInfo VertexInputState;
    vk::PipelineRasterizationStateCreateInfo RasterizationState;
    vk::PipelineInputAssemblyStateCreateInfo InputAssemblyState;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;
};
