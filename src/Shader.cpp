#include "Shader.h"
#include "File.h"

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>

#include <format>
#include <ranges>

using std::views::transform, std::ranges::find_if, std::ranges::to;

#ifdef RELEASE_BUILD
// All files in `src/shaders` are copied to `build/shaders` at build time.
static const fs::path ShadersDir = "Shaders";
#else
static const fs::path ShadersDir = "../src/shaders"; // Relative to `build/`.
#endif

Shaders::Shaders(std::vector<ShaderTypePath> type_paths)
    : Resources(type_paths | transform([](const auto &tp) { return ShaderResource{tp}; }) | to<std::vector>()) {}
Shaders::Shaders(Shaders &&) = default;
Shaders::~Shaders() = default;

Shaders &Shaders::operator=(Shaders &&) = default;

std::vector<vk::PipelineShaderStageCreateInfo> Shaders::CompileAll(vk::Device device) {
    LayoutBindings.clear();
    BindingByName.clear();

    auto stages =
        Resources | transform([this, device](auto &resource) {
            static const shaderc::Compiler compiler;
            static const auto compile_opts = [] {
                shaderc::CompileOptions opts;
                opts.SetGenerateDebugInfo(); // To get resource variable names for linking with their binding.
                opts.SetOptimizationLevel(shaderc_optimization_level_performance);
                return opts;
            }();
            const auto type = resource.TypePath.Type;
            const auto shader_text = File::Read(ShadersDir / resource.TypePath.Path);
            const auto kind = type == ShaderType::eVertex ? shaderc_glsl_vertex_shader : shaderc_glsl_fragment_shader;
            const auto comp_result = compiler.CompileGlslToSpv(shader_text, kind, "", compile_opts);
            if (comp_result.GetCompilationStatus() != shaderc_compilation_status_success) {
                // todo type to string
                throw std::runtime_error(std::format("Failed to compile {} shader: {}", int(type), comp_result.GetErrorMessage()));
            }
            std::vector<uint> spirv_words{comp_result.cbegin(), comp_result.cend()};
            resource.Module = device.createShaderModuleUnique({{}, spirv_words});
            spirv_cross::Compiler comp(std::move(spirv_words));
            resource.Resources = std::make_unique<spirv_cross::ShaderResources>(comp.get_shader_resources());

            for (const auto &resource : resource.Resources->uniform_buffers) {
                // Only using a single set for now. Otherwise, we'd group LayoutBindings by set.
                // uint set = comp.get_decoration(resource.id, spv::DecorationDescriptorSet);

                const uint binding = comp.get_decoration(resource.id, spv::DecorationBinding);
                if (!BindingByName.contains(resource.name)) {
                    BindingByName.emplace(resource.name, binding);
                    // Keep LayoutBindings sorted by binding number.
                    const auto pos = std::lower_bound(LayoutBindings.begin(), LayoutBindings.end(), binding, [](const auto &b, uint i) { return b.binding < i; });
                    LayoutBindings.insert(pos, {binding, vk::DescriptorType::eUniformBuffer, 1, type, nullptr});
                } else {
                    LayoutBindings[binding].stageFlags |= type; // This binding is used in multiple stages.
                }
            }

            for (const auto &resource : resource.Resources->storage_buffers) {
                const uint binding = comp.get_decoration(resource.id, spv::DecorationBinding);
                if (!BindingByName.contains(resource.name)) {
                    BindingByName.emplace(resource.name, binding);
                    const auto pos = std::lower_bound(LayoutBindings.begin(), LayoutBindings.end(), binding, [](const auto &b, uint i) { return b.binding < i; });
                    LayoutBindings.insert(pos, {binding, vk::DescriptorType::eStorageBuffer, 1, type, nullptr});
                } else {
                    LayoutBindings[binding].stageFlags |= type;
                }
            }

            for (const auto &resource : resource.Resources->sampled_images) {
                const uint binding = comp.get_decoration(resource.id, spv::DecorationBinding);
                LayoutBindings.emplace_back(binding, vk::DescriptorType::eCombinedImageSampler, 1, type);
                BindingByName.emplace(resource.name, binding);
            }
            return vk::PipelineShaderStageCreateInfo{vk::PipelineShaderStageCreateFlags{}, type, *resource.Module, "main"};
        }) |
        to<std::vector>();
    return stages;
}

ShaderPipeline::ShaderPipeline(
    vk::Device device, vk::DescriptorPool descriptor_pool, ::Shaders &&shaders,
    vk::PipelineVertexInputStateCreateInfo vertex_input_state,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology,
    vk::PipelineColorBlendAttachmentState color_blend_attachment,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state,
    vk::SampleCountFlagBits msaa_samples,
    std::optional<vk::PushConstantRange> push_constant_range,
    float depth_bias
) : Device(device), Shaders(std::move(shaders)),
    VertexInputState(std::move(vertex_input_state)),
    MultisampleState({{}, msaa_samples}),
    ColorBlendAttachment(std::move(color_blend_attachment)),
    DepthStencilState(std::move(depth_stencil_state)),
    RasterizationState({{}, false, false, polygon_mode, {}, vk::FrontFace::eCounterClockwise, depth_bias != 0.f, depth_bias, {}, {}, 1.f}),
    InputAssemblyState({{}, topology}) {
    Shaders.CompileAll(Device); // Populates descriptor sets. todo This is done redundantly for all shaders in `Compile` at app startup.
    DescriptorSetLayout = Device.createDescriptorSetLayoutUnique({{}, Shaders.GetLayoutBindings()});
    PipelineLayout = Device.createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), push_constant_range ? 1u : 0u, push_constant_range ? &*push_constant_range : nullptr});
    const vk::DescriptorSetAllocateInfo alloc_info{descriptor_pool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique(alloc_info).front());
}

std::optional<vk::WriteDescriptorSet> ShaderPipeline::CreateWriteDescriptorSet(std::string_view binding_name, const vk::DescriptorBufferInfo *buffer_info, const vk::DescriptorImageInfo *image_info) const {
    if (!Shaders.HasBinding(binding_name)) return std::nullopt;

    const auto binding = Shaders.GetBinding(binding_name);
    if (buffer_info) {
        const auto &layout_bindings = Shaders.GetLayoutBindings();
        const auto it = find_if(layout_bindings, [binding](const auto &b) { return b.binding == binding; });
        return {{*DescriptorSet, binding, 0, 1, it->descriptorType, nullptr, buffer_info}};
    }
    return {{*DescriptorSet, binding, 0, 1, vk::DescriptorType::eCombinedImageSampler, image_info, nullptr}};
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
    cb.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *PipelineLayout, 0, 1, &*DescriptorSet, 0, nullptr);
    cb.draw(4, 1, 0, 0); // Draw a full-screen quad triangle strip.
}
