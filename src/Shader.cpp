#include "Shader.h"
#include "File.h"

#include <cassert>
#include <format>
#include <print>
#include <ranges>
#include <shaderc/shaderc.hpp>

using std::views::transform, std::ranges::find_if, std::ranges::to;

#ifdef RELEASE_BUILD
// All files in `src/shaders` are copied to `build/shaders` at build time.
static const fs::path ShadersDir = "shaders";
#else
static const fs::path ShadersDir = "../src/shaders"; // Relative to `build/`.
#endif

// Resolves #include directives relative to `ShadersDir`
class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type, const char *, size_t
    ) override {
        const auto path = ShadersDir / requested_source;
        auto *result = new shaderc_include_result;
        try {
            auto *content = new std::string(File::Read(path));
            auto *name = new std::string(path.string());
            result->source_name = name->c_str();
            result->source_name_length = name->size();
            result->content = content->c_str();
            result->content_length = content->size();
            result->user_data = new IncludeData{content, name};
        } catch (...) {
            auto *error = new std::string(std::format("Failed to include '{}'", requested_source));
            result->source_name = "";
            result->source_name_length = 0;
            result->content = error->c_str();
            result->content_length = error->size();
            result->user_data = new IncludeData{error, nullptr};
        }
        return result;
    }

    void ReleaseInclude(shaderc_include_result *data) override {
        auto *include_data = static_cast<IncludeData *>(data->user_data);
        delete include_data->Content;
        delete include_data->Name;
        delete include_data;
        delete data;
    }

private:
    struct IncludeData {
        std::string *Content;
        std::string *Name;
    };
};

Shaders::Shaders(std::vector<ShaderTypePath> type_paths)
    : Resources(type_paths | transform([](const auto &tp) { return ShaderResource{tp}; }) | to<std::vector>()) {}
Shaders::Shaders(Shaders &&) = default;
Shaders::~Shaders() = default;

Shaders &Shaders::operator=(Shaders &&) = default;

std::vector<vk::PipelineShaderStageCreateInfo> Shaders::CompileAll(vk::Device device) {
    auto stages =
        Resources | transform([device](auto &resource) {
            static const shaderc::Compiler compiler;
            shaderc::CompileOptions compile_opts;
            compile_opts.SetGenerateDebugInfo(); // To get resource variable names for linking with their binding.
            compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);
            compile_opts.SetIncluder(std::make_unique<ShaderIncluder>());
            const auto type = resource.TypePath.Type;
            const auto path = ShadersDir / resource.TypePath.Path;
            const auto kind = [type] {
                switch (type) {
                    case ShaderType::eVertex: return shaderc_glsl_vertex_shader;
                    case ShaderType::eFragment: return shaderc_glsl_fragment_shader;
                    case ShaderType::eCompute: return shaderc_glsl_compute_shader;
                    default: throw std::runtime_error(std::format("Unsupported shader stage {}", vk::to_string(type)));
                }
            }();
            const auto comp_result = compiler.CompileGlslToSpv(File::Read(path), kind, "", compile_opts);
            if (comp_result.GetCompilationStatus() != shaderc_compilation_status_success) {
                const auto error_message = comp_result.GetErrorMessage();
                throw std::runtime_error(std::format("Error compiling shader {}:\n{}", path.string(), error_message));
            }
            std::vector<uint> spirv_words{comp_result.cbegin(), comp_result.cend()};
            resource.Module = device.createShaderModuleUnique({{}, spirv_words});
            return vk::PipelineShaderStageCreateInfo{vk::PipelineShaderStageCreateFlags{}, type, *resource.Module, "main"};
        }) |
        to<std::vector>();
    return stages;
}

ShaderPipeline::ShaderPipeline(
    vk::Device device, ::Shaders &&shaders,
    vk::PipelineVertexInputStateCreateInfo vertex_input_state,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology,
    vk::PipelineColorBlendAttachmentState color_blend_attachment,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state,
    vk::SampleCountFlagBits msaa_samples,
    std::optional<vk::PushConstantRange> push_constant_range,
    float depth_bias,
    vk::DescriptorSetLayout set_layout,
    vk::DescriptorSet set
) : Device(device), Shaders(std::move(shaders)),
    VertexInputState(std::move(vertex_input_state)),
    MultisampleState({{}, msaa_samples}),
    ColorBlendAttachment(std::move(color_blend_attachment)),
    DepthStencilState(std::move(depth_stencil_state)),
    RasterizationState({{}, false, false, polygon_mode, {}, vk::FrontFace::eCounterClockwise, depth_bias != 0.f, depth_bias, {}, {}, 1.f}),
    InputAssemblyState({{}, topology}),
    DescriptorSetLayout(set_layout),
    DescriptorSet(set) {
    assert(DescriptorSetLayout && DescriptorSet && "Bindless descriptor set/layout required.");
    PipelineLayout = Device.createPipelineLayoutUnique({{}, 1, &DescriptorSetLayout, push_constant_range ? 1u : 0u, push_constant_range ? &*push_constant_range : nullptr});
}

void ShaderPipeline::Compile(vk::RenderPass render_pass) {
    const auto shader_stages = Shaders.CompileAll(Device);

    static constexpr vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static constexpr std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};

    const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, 1, &ColorBlendAttachment};
    auto pipeline_result = Device.createGraphicsPipelineUnique(
        {},
        {
            {},
            shader_stages,
            &VertexInputState,
            &InputAssemblyState,
            nullptr,
            &viewport_state,
            &RasterizationState,
            &MultisampleState,
            DepthStencilState.has_value() ? &*DepthStencilState : nullptr,
            &color_blending,
            &dynamic_state,
            *PipelineLayout,
            render_pass,
        }
    );
    if (pipeline_result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to create graphics pipeline: {}", vk::to_string(pipeline_result.result)));
    }
    Pipeline = std::move(pipeline_result.value);
}

void ShaderPipeline::RenderQuad(vk::CommandBuffer cb) const {
    cb.bindPipeline(vk::PipelineBindPoint::eGraphics, *Pipeline);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *PipelineLayout, 0, 1, &DescriptorSet, 0, nullptr);
    cb.draw(4, 1, 0, 0); // Draw a full-screen quad triangle strip.
}

ComputePipeline::ComputePipeline(vk::Device d, Shaders &&shaders, std::optional<vk::PushConstantRange> push_constant_range, vk::DescriptorSetLayout set_layout, vk::DescriptorSet set)
    : Device(d), ShaderModules(std::move(shaders)), DescriptorSetLayout(set_layout), DescriptorSet(set) {
    assert(DescriptorSetLayout && DescriptorSet && "Bindless descriptor set/layout required.");
    PipelineLayout = d.createPipelineLayoutUnique({{}, 1, &DescriptorSetLayout, push_constant_range ? 1u : 0u, push_constant_range ? &*push_constant_range : nullptr});
    Compile();
}

void ComputePipeline::Compile() {
    const auto shader_stages = ShaderModules.CompileAll(Device);
    assert(shader_stages.size() == 1 && shader_stages.front().stage == vk::ShaderStageFlagBits::eCompute && "Compute pipeline expects a single compute shader stage");
    const vk::ComputePipelineCreateInfo pipeline_info{{}, shader_stages.front(), *PipelineLayout};
    Pipeline = Device.createComputePipelineUnique({}, pipeline_info).value;
}

ShaderPipeline PipelineContext::CreateGraphics(
    Shaders &&shaders,
    vk::PipelineVertexInputStateCreateInfo vertex_input,
    vk::PolygonMode polygon_mode,
    vk::PrimitiveTopology topology,
    vk::PipelineColorBlendAttachmentState color_blend,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil,
    std::optional<vk::PushConstantRange> push_constants,
    float depth_bias
) const {
    return {
        Device, std::move(shaders), vertex_input,
        polygon_mode, topology, color_blend, depth_stencil,
        MsaaSamples, push_constants, depth_bias, SharedLayout, SharedSet
    };
}

ComputePipeline PipelineContext::CreateCompute(
    Shaders &&shaders,
    std::optional<vk::PushConstantRange> push_constants
) const {
    return {Device, std::move(shaders), push_constants, SharedLayout, SharedSet};
}
