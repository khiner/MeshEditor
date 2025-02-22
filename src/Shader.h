#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <vector>

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

    // Populates all fields.
    std::vector<vk::PipelineShaderStageCreateInfo> CompileAll(vk::Device);
    std::vector<uint> Compile(ShaderType) const;

    bool HasBinding(std::string_view name) const { return BindingByName.contains(name); }
    uint GetBinding(std::string_view name) const { return BindingByName.at(name); }

    std::unordered_map<ShaderType, fs::path> Paths; // Paths are relative to the `Shaders` directory.
    std::unordered_map<ShaderType, vk::UniqueShaderModule> Modules;
    std::unordered_map<ShaderType, std::unique_ptr<spirv_cross::ShaderResources>> Resources;
    std::vector<vk::DescriptorSetLayoutBinding> LayoutBindings; // Sorted by binding number.
    std::unordered_map<std::string_view, uint> BindingByName;
};

// Convenience generators for default pipeline states.
constexpr vk::PipelineDepthStencilStateCreateInfo CreateDepthStencil(bool test_depth = true, bool write_depth = true, vk::CompareOp depth_compare_op = vk::CompareOp::eLess) {
    return {
        {}, // flags
        test_depth, // depthTestEnable
        write_depth, // depthWriteEnable
        depth_compare_op, // depthCompareOp
        VK_FALSE, // depthBoundsTestEnable
        VK_FALSE, // stencilTestEnable
        {}, // front (stencil state for front faces)
        {}, // back (stencil state for back faces)
        0.f, // minDepthBounds
        1.f // maxDepthBounds
    };
}
constexpr vk::PipelineColorBlendAttachmentState CreateColorBlendAttachment(bool blend = true) {
    if (blend) {
        return {
            true,
            vk::BlendFactor::eSrcAlpha, // srcCol
            vk::BlendFactor::eOneMinusSrcAlpha, // dstCol
            vk::BlendOp::eAdd, // colBlend
            vk::BlendFactor::eOne, // srcAlpha
            vk::BlendFactor::eOne, // dstAlpha
            vk::BlendOp::eAdd, // alphaBlend
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        };
    }
    return {
        false,
        vk::BlendFactor::eOne, // srcCol
        vk::BlendFactor::eZero, // dstCol
        vk::BlendOp::eAdd, // colBlend
        vk::BlendFactor::eOne, // srcAlpha
        vk::BlendFactor::eZero, // dstAlpha
        vk::BlendOp::eAdd, // alphaBlend
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
}

struct ShaderPipeline {
    ShaderPipeline(
        vk::Device, vk::DescriptorPool, Shaders &&,
        vk::PipelineVertexInputStateCreateInfo,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
        vk::PipelineColorBlendAttachmentState color_blend_attachment = {},
        std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state = {},
        vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1
    );
    ShaderPipeline(ShaderPipeline &&) = default;
    ShaderPipeline &operator=(ShaderPipeline &&) = default;
    ~ShaderPipeline() = default;

    vk::Device Device;

    Shaders Shaders;

    vk::PipelineVertexInputStateCreateInfo VertexInputState;
    vk::PipelineMultisampleStateCreateInfo MultisampleState;
    vk::PipelineColorBlendAttachmentState ColorBlendAttachment;
    std::optional<vk::PipelineDepthStencilStateCreateInfo> DepthStencilState;
    vk::PipelineRasterizationStateCreateInfo RasterizationState;
    vk::PipelineInputAssemblyStateCreateInfo InputAssemblyState;

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    void Compile(vk::RenderPass); // Recompile all shaders and update `Pipeline`.

    std::optional<vk::WriteDescriptorSet> CreateWriteDescriptorSet(std::string_view binding_name, const vk::DescriptorBufferInfo *, const vk::DescriptorImageInfo *) const;

    void RenderQuad(vk::CommandBuffer) const;
};
