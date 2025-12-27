#pragma once

#include <vulkan/vulkan.hpp>

#include <filesystem>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

using ShaderType = vk::ShaderStageFlagBits;

struct ShaderTypePath {
    ShaderType Type;
    fs::path Path; // Paths are relative to the `shaders` directory.
};

struct Shaders {
    Shaders(std::vector<ShaderTypePath>);
    Shaders(Shaders &&);
    ~Shaders();

    Shaders &operator=(Shaders &&);

    // Populates modules and returns shader stage infos.
    std::vector<vk::PipelineShaderStageCreateInfo> CompileAll(vk::Device);

private:
    struct ShaderResource {
        ShaderTypePath TypePath;
        vk::UniqueShaderModule Module{};
    };
    std::vector<ShaderResource> Resources;
};

constexpr vk::PipelineDepthStencilStateCreateInfo CreateDepthStencil(bool test = true, bool write = true, vk::CompareOp compare_op = vk::CompareOp::eLess) {
    return {
        {}, // flags
        test, // depthTestEnable
        write, // depthWriteEnable
        compare_op, // depthCompareOp
        vk::False, // depthBoundsTestEnable
        vk::False, // stencilTestEnable
        {}, // front (stencil state for front faces)
        {}, // back (stencil state for back faces)
        0.f, // minDepthBounds
        1.f // maxDepthBounds
    };
}
constexpr vk::PipelineColorBlendAttachmentState CreateColorBlendAttachment(bool blend = true) {
    if (blend) {
        return {
            true, // blendEnable
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
        vk::Device, Shaders &&,
        vk::PipelineVertexInputStateCreateInfo,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
        vk::PipelineColorBlendAttachmentState color_blend_attachment = {},
        std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state = {},
        vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1,
        std::optional<vk::PushConstantRange> push_constant_range = std::nullopt,
        float depth_bias = 0.f,
        vk::DescriptorSetLayout set_layout = {},
        vk::DescriptorSet set = {}
    );
    ShaderPipeline(ShaderPipeline &&) = default;
    ~ShaderPipeline() = default;

    vk::Device Device;

    Shaders Shaders;

    vk::PipelineVertexInputStateCreateInfo VertexInputState;
    vk::PipelineMultisampleStateCreateInfo MultisampleState;
    vk::PipelineColorBlendAttachmentState ColorBlendAttachment;
    std::optional<vk::PipelineDepthStencilStateCreateInfo> DepthStencilState;
    vk::PipelineRasterizationStateCreateInfo RasterizationState;
    vk::PipelineInputAssemblyStateCreateInfo InputAssemblyState;

    vk::DescriptorSetLayout DescriptorSetLayout{};
    vk::DescriptorSet DescriptorSet{};
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    vk::DescriptorSetLayout GetDescriptorSetLayout() const { return DescriptorSetLayout; }
    vk::DescriptorSet GetDescriptorSet() const { return DescriptorSet; }

    void Compile(vk::RenderPass); // Recompile all shaders and update `Pipeline`.

    void RenderQuad(vk::CommandBuffer) const;
};

struct ComputePipeline {
    ComputePipeline(
        vk::Device, Shaders &&, std::optional<vk::PushConstantRange> = std::nullopt,
        vk::DescriptorSetLayout set_layout = {}, vk::DescriptorSet set = {}
    );
    ComputePipeline(ComputePipeline &&) = default;
    ~ComputePipeline() = default;

    vk::Device Device;
    Shaders ShaderModules;

    vk::DescriptorSetLayout DescriptorSetLayout{};
    vk::DescriptorSet DescriptorSet{};
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    vk::DescriptorSetLayout GetDescriptorSetLayout() const { return DescriptorSetLayout; }
    vk::DescriptorSet GetDescriptorSet() const { return DescriptorSet; }

    void Compile();
};

struct PipelineContext {
    vk::Device Device;
    vk::DescriptorSetLayout SharedLayout;
    vk::DescriptorSet SharedSet;
    vk::SampleCountFlagBits MsaaSamples;

    ShaderPipeline CreateGraphics(
        Shaders &&shaders,
        vk::PipelineVertexInputStateCreateInfo vertex_input,
        vk::PolygonMode polygon_mode,
        vk::PrimitiveTopology topology,
        vk::PipelineColorBlendAttachmentState color_blend,
        std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil,
        std::optional<vk::PushConstantRange> push_constants = std::nullopt,
        float depth_bias = 0.f
    ) const;

    ComputePipeline CreateCompute(
        Shaders &&shaders,
        std::optional<vk::PushConstantRange> push_constants = std::nullopt
    ) const;
};
