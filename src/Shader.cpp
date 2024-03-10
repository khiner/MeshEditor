#include "Shader.h"

#include <format>

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>

#include "numeric/mat4.h"

#include "File.h"
#include "mesh/Vertex.h"

#ifdef DEBUG_BUILD
static const fs::path ShadersDir = "../src/shaders"; // Relative to `build/`.
#elif defined(RELEASE_BUILD)
// All files in `src/shaders` are copied to `build/shaders` at build time.
static const fs::path ShadersDir = "Shaders";
#endif

Shaders::Shaders(std::unordered_map<ShaderType, fs::path> &&paths) : Paths(std::move(paths)) {}
Shaders::Shaders(Shaders &&) = default;
Shaders::~Shaders() = default;
Shaders &Shaders::operator=(Shaders &&) = default;

std::vector<uint> Shaders::Compile(ShaderType type) const {
    if (!Paths.contains(type)) throw std::runtime_error(std::format("No path for shader type: {}", int(type)));

    static const shaderc::Compiler compiler;
    static shaderc::CompileOptions compile_opts;
    compile_opts.SetGenerateDebugInfo(); // To get resource variable names for linking with their binding.
    compile_opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc_shader_kind kind = type == ShaderType::eVertex ? shaderc_glsl_vertex_shader : shaderc_glsl_fragment_shader;
    const std::string path = Paths.at(type);
    const std::string shader_text = File::Read(ShadersDir / path);

    const auto spirv = compiler.CompileGlslToSpv(shader_text, kind, "", compile_opts);
    if (spirv.GetCompilationStatus() != shaderc_compilation_status_success) {
        // todo type to string
        throw std::runtime_error(std::format("Failed to compile {} shader: {}", int(type), spirv.GetErrorMessage()));
    }
    return {spirv.cbegin(), spirv.cend()};
}

std::vector<vk::PipelineShaderStageCreateInfo> Shaders::CompileAll(vk::Device device) {
    Modules.clear();
    Resources.clear();
    LayoutBindings.clear();
    BindingByName.clear();

    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.reserve(Paths.size());
    for (const auto &[type, path] : Paths) {
        const auto &spirv = Compile(type);
        spirv_cross::Compiler comp(spirv);
        Resources[type] = std::make_unique<spirv_cross::ShaderResources>(comp.get_shader_resources());
        Modules[type] = device.createShaderModuleUnique({{}, spirv});

        for (const auto &resource : Resources.at(type)->uniform_buffers) {
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

        for (const auto &resource : Resources.at(type)->sampled_images) {
            const uint binding = comp.get_decoration(resource.id, spv::DecorationBinding);
            LayoutBindings.emplace_back(binding, vk::DescriptorType::eCombinedImageSampler, 1, type);
            BindingByName.emplace(resource.name, binding);
        }

        stages.push_back({vk::PipelineShaderStageCreateFlags{}, type, *Modules.at(type), "main"});
    }
    return stages;
}

namespace Format {
const auto Vec3 = vk::Format::eR32G32B32Sfloat;
const auto Vec4 = vk::Format::eR32G32B32A32Sfloat;
} // namespace Format

vk::PipelineVertexInputStateCreateInfo GenerateVertex3DInputState() {
    static const std::vector<vk::VertexInputBindingDescription> bindings{
        {0, sizeof(Vertex3D), vk::VertexInputRate::eVertex},
        {1, sizeof(mat4), vk::VertexInputRate::eInstance}
    };
    static const std::vector<vk::VertexInputAttributeDescription> attrs{
        {0, 0, Format::Vec3, offsetof(Vertex3D, Position)},
        {1, 0, Format::Vec3, offsetof(Vertex3D, Normal)},
        {2, 0, Format::Vec4, offsetof(Vertex3D, Color)},
        // Instance mat4, one vec4 per row
        {3, 1, Format::Vec4, 0},
        {4, 1, Format::Vec4, sizeof(vec4)},
        {5, 1, Format::Vec4, 2 * sizeof(vec4)},
        {6, 1, Format::Vec4, 3 * sizeof(vec4)}
    };
    return {{}, bindings, attrs};
}

ShaderPipeline::ShaderPipeline(
    vk::Device device, vk::DescriptorPool descriptor_pool, ::Shaders &&shaders,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology,
    vk::PipelineColorBlendAttachmentState color_blend_attachment,
    std::optional<vk::PipelineDepthStencilStateCreateInfo> depth_stencil_state,
    vk::SampleCountFlagBits msaa_samples
) : Device(device), Shaders(std::move(shaders)),
    MultisampleState({{}, msaa_samples}),
    ColorBlendAttachment(std::move(color_blend_attachment)),
    DepthStencilState(std::move(depth_stencil_state)),
    VertexInputState(GenerateVertex3DInputState()),
    RasterizationState({{}, false, false, polygon_mode, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.f}),
    InputAssemblyState({{}, topology}) {
    Shaders.CompileAll(Device); // Populates descriptor sets. todo This is done redundantly for all shaders in `Compile` at app startup.
    DescriptorSetLayout = Device.createDescriptorSetLayoutUnique({{}, Shaders.LayoutBindings});
    PipelineLayout = Device.createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), 0});
    const vk::DescriptorSetAllocateInfo alloc_info{descriptor_pool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(Device.allocateDescriptorSetsUnique(alloc_info).front());
}

std::optional<vk::WriteDescriptorSet> ShaderPipeline::CreateWriteDescriptorSet(const std::string &binding_name, const vk::DescriptorBufferInfo *buffer_info, const vk::DescriptorImageInfo *image_info) const {
    if (!Shaders.HasBinding(binding_name)) return std::nullopt;

    const auto binding = Shaders.GetBinding(binding_name);
    if (buffer_info) return {{*DescriptorSet, binding, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, buffer_info}};
    return {{*DescriptorSet, binding, 0, 1, vk::DescriptorType::eCombinedImageSampler, image_info, nullptr}};
}

void ShaderPipeline::Compile(vk::RenderPass render_pass) {
    const auto shader_stages = Shaders.CompileAll(Device);

    static const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
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
