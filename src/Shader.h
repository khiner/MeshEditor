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

struct ShaderTypePath {
    ShaderType Type;
    fs::path Path; // Paths are relative to the `Shaders` directory.
};

struct Shaders {
    Shaders(std::vector<ShaderTypePath>);
    Shaders(Shaders &&);
    ~Shaders();

    Shaders &operator=(Shaders &&);

    // Populates all fields.
    std::vector<vk::PipelineShaderStageCreateInfo> CompileAll(vk::Device);

    bool HasBinding(std::string_view name) const { return BindingByName.contains(name); }
    uint GetBinding(std::string_view name) const { return BindingByName.at(name); }
    const auto &GetLayoutBindings() const { return LayoutBindings; }

private:
    std::vector<vk::DescriptorSetLayoutBinding> LayoutBindings{}; // Sorted by binding number.
    std::unordered_map<std::string_view, uint> BindingByName{};

    struct ShaderResource {
        ShaderTypePath TypePath;
        vk::UniqueShaderModule Module{};
        std::unique_ptr<spirv_cross::ShaderResources> Resources{};
    };
    std::vector<ShaderResource> Resources;
};

// Convenience generators for default pipeline states.
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
        vk::Device, vk::DescriptorPool, Shaders &&,
        vk::PipelineVertexInputStateCreateInfo,
        vk::PolygonMode polygon_mode = vk::PolygonMode::eFill,
        vk::PrimitiveTopology topology = vk::PrimitiveTopology::eTriangleList,
        vk::PipelineColorBlendAttachmentState color_blend_attachment = {},
        std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state = {},
        vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1,
        std::optional<vk::PushConstantRange> push_constant_range = std::nullopt,
        float depth_bias = 0.f
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

    vk::UniqueDescriptorSetLayout DescriptorSetLayout;
    vk::UniqueDescriptorSet DescriptorSet;
    vk::UniquePipelineLayout PipelineLayout;
    vk::UniquePipeline Pipeline;

    void Compile(vk::RenderPass); // Recompile all shaders and update `Pipeline`.

    std::optional<vk::WriteDescriptorSet> CreateWriteDescriptorSet(std::string_view binding_name, const vk::DescriptorBufferInfo *, const vk::DescriptorImageInfo *) const;

    void RenderQuad(vk::CommandBuffer) const;
};
