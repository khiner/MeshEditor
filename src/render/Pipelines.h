#pragma once

#include "render/IblPrefilterPipelines.h"
#include "render/PbrFeature.h"
#include "render/ShaderPipelineType.h"
#include "vulkan/Image.h"

#include <unordered_map>

using SPT = ShaderPipelineType;

inline constexpr vk::ImageSubresourceRange DepthSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
inline constexpr vk::ImageSubresourceRange ColorSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders();
    const ShaderPipeline &Bind(vk::CommandBuffer, SPT) const;
};

namespace Format {
inline constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
inline constexpr auto HdrColor = vk::Format::eR16G16B16A16Sfloat;
inline constexpr auto Depth = vk::Format::eD32Sfloat;
inline constexpr auto Float = vk::Format::eR32Sfloat;
inline constexpr auto Float2 = vk::Format::eR32G32Sfloat;
inline constexpr auto LineData = vk::Format::eR8G8B8A8Unorm;
inline constexpr auto Uint = vk::Format::eR32Uint;
} // namespace Format

// Compiles a PBR pipeline shaped to the scene's active feature set via Vulkan specialization constants.
// Vert + frag VkShaderModules are compiled once; only vkCreateGraphicsPipeline runs on mask changes.
struct PbrCompiler {
    PbrCompiler(PipelineContext, vk::RenderPass, vk::RenderPass prepass_render_pass);

    enum class Variant {
        Opaque,
        Blend,
        OpaquePrepass
    };

    bool CompilePipelines(PbrFeatureMask);
    vk::PipelineLayout Bind(vk::CommandBuffer, Variant) const;
    bool HasFeature(PbrFeature f) const { return ::HasFeature(Mask, f); }
    void RecompileModules(); // hot reload: recompile SPIRV and pipelines from disk

private:
    void CompileModules();
    vk::UniquePipeline CreateTargetedPipeline(const vk::SpecializationInfo &frag_spec, bool depth_write, vk::RenderPass) const;

    vk::Device Device;
    vk::UniquePipelineCache Cache;
    vk::UniqueShaderModule VertModule, FragModule;
    vk::UniquePipelineLayout Layout;
    vk::DescriptorSetLayout SetLayout;
    vk::DescriptorSet Set;
    vk::RenderPass RenderPass, PrepassRenderPass;

    PbrFeatureMask Mask{0};
    vk::UniquePipeline OpaqueTargeted, BlendTargeted, OpaquePrepass;
};

struct MainPipeline {
    MainPipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {});

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass, vk::RenderPass line_aa_render_pass);

        mvk::ImageResource DepthImage, ColorImage, LineDataImage, FinalColorImage;
        vk::UniqueSampler NearestSampler;
        vk::UniqueFramebuffer Framebuffer, LineAAFramebuffer;
    };

    // Mip chain + framebuffer backing real-transmission sampling.
    // Scene-linear HDR (no exposure/tone map/sRGB), so the main pass samples true radiance.
    // Allocated on demand only when a scene material uses KHR_materials_transmission and the user toggle is on.
    struct TransmissionResourcesT {
        TransmissionResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass, vk::ImageView depth_view, vk::ImageView line_data_view);

        mvk::ImageResource Image;
        vk::UniqueImageView Mip0View;
        uint32_t MipCount{1};
        vk::Extent2D Extent{};
        vk::UniqueSampler Sampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice);
    // Idempotent: allocates if `wanted` and not already at the right extent; releases if not wanted.
    // Returns true when the allocated state changed (caller should re-write the descriptor write).
    bool EnsureTransmissionResources(vk::Extent2D, vk::Device, vk::PhysicalDevice, bool wanted);

    PipelineRenderer Renderer;
    // Transmission pre-pass: same attachment layout as Renderer but an HDR color target.
    vk::UniqueRenderPass PrepassRenderPass;
    ShaderPipeline PrepassBackground; // Background variant outputting scene-linear radiance
    vk::UniqueRenderPass LineAARenderPass;
    ShaderPipeline LineAAComposite;
    std::unique_ptr<ResourcesT> Resources;
    std::unique_ptr<TransmissionResourcesT> Transmission;

    PbrCompiler Compiler;
};

struct SilhouettePipeline {
    SilhouettePipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {});

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass);

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice);

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SilhouetteEdgePipeline {
    SilhouetteEdgePipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {});

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass);

        mvk::ImageResource DepthImage, OffscreenImage;
        vk::UniqueSampler ImageSampler, DepthSampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice);

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct SelectionFragmentPipeline {
    // Render pass that loads depth from silhouette pass for element occlusion.
    SelectionFragmentPipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {});

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass, vk::ImageView silhouette_depth_view);

        mvk::ImageResource HeadImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::ImageView silhouette_depth_view);

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
};

struct Pipelines {
    Pipelines(vk::Device, vk::PhysicalDevice, vk::DescriptorSetLayout selection_layout = {}, vk::DescriptorSet selection_set = {});

    vk::Device Device;
    vk::PhysicalDevice PhysicalDevice;
    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    ComputePipeline ObjectPick, ElementPick, BoxSelect, UpdateSelectionState;
    IblPrefilterPipelines IblPrefilter;

    void SetExtent(vk::Extent2D);
    void CompileShaders();

    // Zero before render resources exist.
    vk::Extent3D BuiltColorExtent() const { return Main.Resources ? Main.Resources->ColorImage.Extent : vk::Extent3D{}; }
};
