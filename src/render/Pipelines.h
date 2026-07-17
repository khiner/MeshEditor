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
    const ShaderPipeline &Bind(vk::CommandBuffer, SPT, uint32_t scene_ubo_offset = 0) const;
};

namespace Format {
inline constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
inline constexpr auto HdrColor = vk::Format::eR16G16B16A16Sfloat;
inline constexpr auto Depth = vk::Format::eD32Sfloat;
inline constexpr auto Float = vk::Format::eR32Sfloat;
inline constexpr auto Float2 = vk::Format::eR32G32Sfloat;
inline constexpr auto LineData = vk::Format::eR8G8B8A8Unorm;
inline constexpr auto Velocity = vk::Format::eR16G16B16A16Sfloat; // (prev->current, current->next) screen motion
inline constexpr auto Uint = vk::Format::eR32Uint;
} // namespace Format

// Compiles a PBR pipeline shaped to the scene's active feature set via Vulkan specialization constants.
// VkShaderModules are compiled once. Only vkCreateGraphicsPipeline runs on mask changes.
// The velocity variants render motion blur steps: they target the scene+velocity render pass, with
// opaque geometry writing its screen motion from VELOCITY_OUTPUT shader modules.
struct PbrCompiler {
    PbrCompiler(PipelineContext, vk::RenderPass scene, vk::RenderPass scene_velocity);

    enum class Variant {
        Opaque,
        Blend,
        OpaqueVelocity,
        BlendVelocity,
        OpaquePrepass
    };

    bool CompilePipelines(PbrFeatureMask);
    vk::PipelineLayout Bind(vk::CommandBuffer, Variant, uint32_t scene_ubo_offset = 0) const;
    bool HasFeature(PbrFeature f) const { return ::HasFeature(Mask, f); }
    void RecompileModules(); // hot reload: recompile SPIRV and pipelines from disk

private:
    void CompileModules();
    vk::UniquePipeline CreateTargetedPipeline(const vk::SpecializationInfo &frag_spec, bool depth_write, Variant) const;

    vk::Device Device;
    vk::UniquePipelineCache Cache;
    vk::UniqueShaderModule VertModule, FragModule, VelocityVertModule, VelocityFragModule;
    vk::UniquePipelineLayout Layout;
    vk::DescriptorSetLayout SetLayout;
    vk::DescriptorSet Set;
    vk::DescriptorSetLayout UboSetLayout;
    vk::DescriptorSet UboSet;
    vk::RenderPass RenderPass, VelocityRenderPass;

    PbrFeatureMask Mask{0};
    vk::UniquePipeline OpaqueTargeted, BlendTargeted, OpaqueVelocityTargeted, BlendVelocityTargeted, OpaquePrepass;
};

struct MainPipeline {
    MainPipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}, vk::DescriptorSetLayout ubo_layout = {}, vk::DescriptorSet ubo_set = {});

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass scene_render_pass, vk::RenderPass overlay_render_pass, vk::RenderPass composite_render_pass);

        // SceneColorImage holds the shaded scene. OverlayColorImage holds display-referred overlays
        // over transparent, merged onto the scene in the viewport composite.
        mvk::ImageResource DepthImage, SceneColorImage, OverlayColorImage, LineDataImage, FinalColorImage;
        vk::UniqueSampler NearestSampler;
        vk::UniqueFramebuffer SceneFramebuffer, OverlayFramebuffer, CompositeFramebuffer;
    };

    // Mip chain + framebuffer backing real-transmission sampling.
    // Holds un-exposed radiance, which the main pass exposes after sampling.
    // Allocated on demand only when a scene material uses KHR_materials_transmission and the user toggle is on.
    struct TransmissionResourcesT {
        TransmissionResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass, vk::ImageView depth_view);

        mvk::ImageResource Image;
        vk::UniqueImageView Mip0View;
        uint32_t MipCount{1};
        vk::Extent2D Extent{};
        vk::UniqueSampler Sampler;
        vk::UniqueFramebuffer Framebuffer;
    };

    // Motion blur's own targets: the screen motion the gather walks, and the radiance summed across
    // steps. Allocated on demand only while motion blur is active, and sampled through
    // ResourcesT::NearestSampler. SceneVelocityFramebuffer borrows ResourcesT's depth and scene color views.
    struct MotionBlurResourcesT {
        MotionBlurResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::RenderPass accum_render_pass, vk::RenderPass scene_velocity_render_pass, vk::RenderPass gather_render_pass, vk::ImageView depth_view, vk::ImageView scene_color_view);

        // TileImage holds each 32x32 tile's largest motion, GatherImage the blurred scene.
        mvk::ImageResource AccumImage, VelocityImage, TileImage, GatherImage;
        vk::Extent2D TileExtent{};
        vk::UniqueFramebuffer Framebuffer, SceneVelocityFramebuffer, GatherFramebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice);
    // Idempotent: allocates if `wanted` and not already at the right extent, and releases if not wanted.
    // Returns true when the allocated state changed (caller should re-write the descriptor write).
    bool EnsureTransmissionResources(vk::Extent2D, vk::Device, vk::PhysicalDevice, bool wanted);
    // Allocate the motion blur targets at the current color extent if absent. Returns true when they were allocated.
    bool EnsureMotionBlurResources(vk::Device, vk::PhysicalDevice);

    vk::DescriptorImageInfo SceneColorSamplerInfo() const;
    vk::DescriptorImageInfo OverlayColorSamplerInfo() const;
    // Transmission and motion blur targets are lazy. Both fall back to the scene color when
    // unallocated so the binding stays valid. Nothing samples them in that state.
    vk::DescriptorImageInfo TransmissionSamplerInfo() const;
    vk::DescriptorImageInfo MotionBlurAccumSamplerInfo() const;
    vk::DescriptorImageInfo VelocitySamplerInfo() const;
    vk::DescriptorImageInfo SceneDepthSamplerInfo() const;
    // Storage image the motion blur compute passes write. Only valid while MotionBlur is allocated.
    vk::DescriptorImageInfo MotionBlurTileImageInfo() const;
    vk::DescriptorImageInfo MotionBlurGatherSamplerInfo() const;

    // Scene: depth + scene-linear HDR color. Overlays: the scene's depth loaded for occlusion,
    // plus a display-referred overlay color target and the line data driving its AA.
    PipelineRenderer SceneRenderer, OverlayRenderer;
    // Scene pass variant that loads the transmission prepass's depth instead of clearing, for the
    // composite path. Compatible with SceneRenderer's pipelines and framebuffer.
    vk::UniqueRenderPass SceneDepthLoadRenderPass;
    // Scene pass variant for motion blur steps, with a velocity attachment the geometry writes its
    // screen motion into. Holds the fullscreen-quad twins the blurred scene pass binds.
    PipelineRenderer SceneVelocityRenderer;
    // Background variant that skips exposure, for the transmission pre-pass.
    ShaderPipeline PrepassBackground;
    vk::UniqueRenderPass CompositeRenderPass;
    ShaderPipeline ViewportComposite;
    // Single HDR attachment, additive blend. The first step clears the target, so it starts from
    // its own value alone. Later steps load what the earlier ones summed.
    vk::UniqueRenderPass MotionBlurAccumClearRenderPass, MotionBlurAccumRenderPass;
    ShaderPipeline MotionBlurAccumulate;
    // Fullscreen pass that blurs the scene along its screen motion into GatherImage, whose
    // attachment write keeps the output compressed for the accumulate and composite reads.
    vk::UniqueRenderPass MotionBlurGatherRenderPass;
    ShaderPipeline MotionBlurGather;
    std::unique_ptr<ResourcesT> Resources;
    std::unique_ptr<TransmissionResourcesT> Transmission;
    std::unique_ptr<MotionBlurResourcesT> MotionBlur;

    PbrCompiler Compiler;
};

struct SilhouettePipeline {
    SilhouettePipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}, vk::DescriptorSetLayout ubo_layout = {}, vk::DescriptorSet ubo_set = {});

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
    SilhouetteEdgePipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}, vk::DescriptorSetLayout ubo_layout = {}, vk::DescriptorSet ubo_set = {});

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
    SelectionFragmentPipeline(vk::Device, vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}, vk::DescriptorSetLayout ubo_layout = {}, vk::DescriptorSet ubo_set = {});

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
    Pipelines(vk::Device, vk::PhysicalDevice, vk::DescriptorSetLayout selection_layout = {}, vk::DescriptorSet selection_set = {}, vk::DescriptorSetLayout ubo_layout = {}, vk::DescriptorSet ubo_set = {});

    vk::Device Device;
    vk::PhysicalDevice PhysicalDevice;
    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    ComputePipeline ObjectPick, ElementPick, BoxSelect, UpdateSelectionState;
    // Motion blur tile reduction: reduce motion to tiles, then spread each tile's motion over the
    // tiles it crosses. Main.MotionBlurGather blurs the scene along the result.
    ComputePipeline MotionBlurTilesFlatten, MotionBlurTilesDilate;
    IblPrefilterPipelines IblPrefilter;

    void SetExtent(vk::Extent2D);
    void CompileShaders();

    // Zero before render resources exist.
    vk::Extent3D BuiltColorExtent() const { return Main.Resources ? Main.Resources->SceneColorImage.Extent : vk::Extent3D{}; }
};
