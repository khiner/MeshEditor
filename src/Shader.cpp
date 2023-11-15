#include "Shader.h"

#include <format>

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>

#include "File.h"
#include "Vertex.h"

#ifdef DEBUG_BUILD
static const fs::path ShadersDir = "../src/Shaders"; // Relative to `build/`.
#elif defined(RELEASE_BUILD)
// All files in `src/Shaders` are copied to `build/Shaders` at build time.
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

std::vector<vk::PipelineShaderStageCreateInfo> Shaders::CompileAll(const vk::UniqueDevice &device) {
    std::vector<vk::PipelineShaderStageCreateInfo> stages;
    stages.reserve(Paths.size());
    Bindings.clear();
    for (const auto &[type, path] : Paths) {
        const auto &spirv = Compile(type);
        spirv_cross::Compiler comp(spirv);
        Resources[type] = std::make_unique<spirv_cross::ShaderResources>(comp.get_shader_resources());
        Modules[type] = device->createShaderModuleUnique({{}, spirv});

        for (const auto &resource : Resources.at(type)->uniform_buffers) {
            // Only using a single set for now. Otherwise, we'd group bindings by set.
            // uint set = comp.get_decoration(resource.id, spv::DecorationDescriptorSet);
            uint binding = comp.get_decoration(resource.id, spv::DecorationBinding);
            Bindings.emplace_back(binding, vk::DescriptorType::eUniformBuffer, 1, type);
            BindingForResourceName[resource.name] = binding;
        }

        stages.push_back({{}, type, *Modules.at(type), "main"});
    }
    return stages;
}

static vk::PipelineVertexInputStateCreateInfo CreateVertex3DInputState() {
    static const vk::VertexInputBindingDescription vertex_binding{0, sizeof(Vertex3D), vk::VertexInputRate::eVertex};
    static const std::vector<vk::VertexInputAttributeDescription> vertex_attrs{
        {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex3D, Position)},
        {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex3D, Normal)},
        {2, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(Vertex3D, Color)},
    };
    return {{}, vertex_binding, vertex_attrs};
}

ShaderPipeline::ShaderPipeline(
    const vk::UniqueDevice &device, const vk::UniqueDescriptorPool &descriptor_pool, ::Shaders &&shaders,
    vk::PolygonMode polygon_mode, vk::PrimitiveTopology topology,
    bool test_depth, bool write_depth, vk::SampleCountFlagBits msaa_samples
) : Device(device), Shaders(std::move(shaders)),
    MultisampleState({{}, msaa_samples}),
    ColorBlendAttachment{
        true,
        vk::BlendFactor::eSrcAlpha, // srcCol
        vk::BlendFactor::eOneMinusSrcAlpha, // dstCol
        vk::BlendOp::eAdd, // colBlend
        vk::BlendFactor::eOne, // srcAlpha
        vk::BlendFactor::eOne, // dstAlpha
        vk::BlendOp::eAdd, // alphaBlend
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    },
    DepthStencilState({
        {}, // flags
        test_depth, // depthTestEnable
        write_depth, // depthWriteEnable
        vk::CompareOp::eLess, // depthCompareOp
        VK_FALSE, // depthBoundsTestEnable
        VK_FALSE, // stencilTestEnable
        {}, // front (stencil state for front faces)
        {}, // back (stencil state for back faces)
        0.f, // minDepthBounds
        1.f // maxDepthBounds
    }),
    VertexInputState(CreateVertex3DInputState()),
    RasterizationState({{}, false, false, polygon_mode, {}, vk::FrontFace::eCounterClockwise, {}, {}, {}, {}, 1.f}),
    InputAssemblyState({{}, topology}) {
    Shaders.CompileAll(Device); // todo needed populates descriptor sets. This is done redundantly for all shaders in `Compile` at app startup.
    DescriptorSetLayout = Device->createDescriptorSetLayoutUnique({{}, Shaders.Bindings});
    PipelineLayout = Device->createPipelineLayoutUnique({{}, 1, &(*DescriptorSetLayout), 0});
    const vk::DescriptorSetAllocateInfo alloc_info{*descriptor_pool, 1, &(*DescriptorSetLayout)};
    DescriptorSet = std::move(Device->allocateDescriptorSetsUnique(alloc_info).front());
}

void ShaderPipeline::Compile(const vk::UniqueRenderPass &render_pass) {
    const auto shader_stages = Shaders.CompileAll(Device);

    static const vk::PipelineViewportStateCreateInfo viewport_state{{}, 1, nullptr, 1, nullptr};
    static const vk::PipelineColorBlendStateCreateInfo color_blending{{}, false, vk::LogicOp::eCopy, 1, &ColorBlendAttachment};
    static const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    static const vk::PipelineDynamicStateCreateInfo dynamic_state{{}, dynamic_states};
    auto pipeline_result = Device->createGraphicsPipelineUnique(
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
            &DepthStencilState,
            &color_blending,
            &dynamic_state,
            *PipelineLayout,
            *render_pass,
        }
    );
    if (pipeline_result.result != vk::Result::eSuccess) {
        throw std::runtime_error(std::format("Failed to create graphics pipeline: {}", vk::to_string(pipeline_result.result)));
    }
    Pipeline = std::move(pipeline_result.value);
}
