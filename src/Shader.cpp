#include "Shader.h"
#include "File.h"
#include "Paths.h"

#include <cassert>
#include <format>
#include <print>
#include <ranges>
#include <shaderc/shaderc.hpp>

using std::views::transform, std::ranges::find_if, std::ranges::to;

static std::filesystem::path ResolveIncludePath(const std::filesystem::path &requested) {
    auto candidate = Paths::Shaders() / requested;
    std::error_code ec;
    if (std::filesystem::exists(candidate, ec)) return candidate;
    throw std::runtime_error(std::format("Failed to resolve shader path '{}'", requested.string()));
}

class ShaderIncluder : public shaderc::CompileOptions::IncluderInterface {
public:
    shaderc_include_result *GetInclude(
        const char *requested_source, shaderc_include_type, const char *, size_t
    ) override {
        auto *result = new shaderc_include_result;
        try {
            const auto resolved_path = ResolveIncludePath(requested_source);
            auto *include = new IncludeData{File::Read(resolved_path), resolved_path.string()};
            result->source_name = include->Name.c_str();
            result->source_name_length = include->Name.size();
            result->content = include->Content.c_str();
            result->content_length = include->Content.size();
            result->user_data = include;
        } catch (...) {
            auto *include = new IncludeData{std::format("Failed to include '{}'", requested_source), ""};
            result->source_name = "";
            result->source_name_length = 0;
            result->content = include->Content.c_str();
            result->content_length = include->Content.size();
            result->user_data = include;
        }
        return result;
    }

    void ReleaseInclude(shaderc_include_result *data) override {
        delete static_cast<IncludeData *>(data->user_data);
        delete data;
    }

private:
    struct IncludeData {
        std::string Content;
        std::string Name;
    };
};

Shaders::Shaders(std::vector<ShaderTypePath> type_paths)
    : Resources(type_paths | transform([](const auto &tp) {
                    ShaderResource resource{tp};
                    if (!tp.SpecializationConstants.empty()) {
                        ShaderResource::SpecializationData spec{};
                        spec.Entries.reserve(tp.SpecializationConstants.size());
                        spec.Data.reserve(tp.SpecializationConstants.size());
                        size_t offset = 0;
                        for (const auto &[id, value] : tp.SpecializationConstants) {
                            spec.Entries.emplace_back(vk::SpecializationMapEntry{id, static_cast<uint32_t>(offset), sizeof(uint32_t)});
                            spec.Data.push_back(value);
                            offset += sizeof(uint32_t);
                        }
                        spec.Info = vk::SpecializationInfo{
                            static_cast<uint32_t>(spec.Entries.size()),
                            spec.Entries.data(),
                            spec.Data.size() * sizeof(uint32_t),
                            spec.Data.data(),
                        };
                        resource.Specialization = std::move(spec);
                    }
                    return resource;
                }) |
                to<std::vector>()) {}
Shaders::Shaders(Shaders &&) = default;
Shaders::~Shaders() = default;

Shaders &Shaders::operator=(Shaders &&) = default;

static vk::UniqueShaderModule CompileToModule(vk::Device device, ShaderType type, const std::filesystem::path &path) {
    static const shaderc::Compiler compiler;
    shaderc::CompileOptions compile_opts;
    compile_opts.SetGenerateDebugInfo();
    compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);
    compile_opts.SetIncluder(std::make_unique<ShaderIncluder>());
    const auto kind = [type] {
        switch (type) {
            case ShaderType::eVertex: return shaderc_glsl_vertex_shader;
            case ShaderType::eFragment: return shaderc_glsl_fragment_shader;
            case ShaderType::eCompute: return shaderc_glsl_compute_shader;
            default: throw std::runtime_error(std::format("Unsupported shader stage {}", vk::to_string(type)));
        }
    }();
    const auto result = compiler.CompileGlslToSpv(File::Read(path), kind, "", compile_opts);
    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::format("Error compiling shader {}:\n{}", path.string(), result.GetErrorMessage()));
    }
    const std::vector<uint32_t> spirv{result.cbegin(), result.cend()};
    return device.createShaderModuleUnique({{}, spirv});
}

vk::UniqueShaderModule CompileShaderModule(vk::Device device, ShaderType type, const std::filesystem::path &relative_path) {
    return CompileToModule(device, type, Paths::Shaders() / relative_path);
}

std::vector<vk::PipelineShaderStageCreateInfo> Shaders::CompileAll(vk::Device device) {
    auto stages =
        Resources | transform([device](auto &resource) {
            const auto type = resource.TypePath.Type;
            const auto path = Paths::Shaders() / resource.TypePath.Path;
            resource.Module = CompileToModule(device, type, path);
            vk::PipelineShaderStageCreateInfo stage_info{vk::PipelineShaderStageCreateFlags{}, type, *resource.Module, "main"};
            if (resource.Specialization) stage_info.setPSpecializationInfo(&resource.Specialization->Info);
            return stage_info;
        }) |
        to<std::vector>();
    return stages;
}

ShaderPipeline::ShaderPipeline(
    vk::Device device, ::Shaders &&shaders,
    vk::PipelineVertexInputStateCreateInfo vertex_input_state,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology,
    std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state,
    std::optional<vk::PushConstantRange> push_constant_range,
    float depth_bias,
    vk::DescriptorSetLayout set_layout,
    vk::DescriptorSet set
) : Device(device), Shaders(std::move(shaders)),
    VertexInputState(std::move(vertex_input_state)),
    ColorBlendAttachments(std::move(color_blend_attachments)),
    DepthStencilState(std::move(depth_stencil_state)),
    RasterizationState({{}, false, false, polygon_mode, {}, vk::FrontFace::eClockwise, depth_bias != 0.f, depth_bias, {}, depth_bias, 1.f}),
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
    static constexpr vk::PipelineMultisampleStateCreateInfo multisample_state{{}, vk::SampleCountFlagBits::e1};

    const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, static_cast<uint32_t>(ColorBlendAttachments.size()), ColorBlendAttachments.data()};
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
            &multisample_state,
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
    std::vector<vk::PipelineColorBlendAttachmentState> color_blend_attachments,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil,
    std::optional<vk::PushConstantRange> push_constants,
    float depth_bias
) const {
    return {
        Device, std::move(shaders), vertex_input,
        polygon_mode, topology, std::move(color_blend_attachments), depth_stencil,
        push_constants, depth_bias, SharedLayout, SharedSet
    };
}

ComputePipeline PipelineContext::CreateCompute(
    Shaders &&shaders,
    std::optional<vk::PushConstantRange> push_constants
) const {
    return {Device, std::move(shaders), push_constants, SharedLayout, SharedSet};
}
