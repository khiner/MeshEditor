#pragma once

#include "IblPrefilterPipelines.h"
#include "ShaderPipelineType.h"
#include "vulkan/Image.h"

#include <memory>
#include <unordered_map>

using SPT = ShaderPipelineType;

inline constexpr vk::ImageSubresourceRange DepthSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
inline constexpr vk::ImageSubresourceRange ColorSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

struct PipelineRenderer {
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, ShaderPipeline> ShaderPipelines;

    void CompileShaders();
    const ShaderPipeline &Bind(vk::CommandBuffer cb, SPT spt) const;
};

namespace Format {
inline constexpr auto Color = vk::Format::eB8G8R8A8Unorm;
inline constexpr auto Depth = vk::Format::eD32Sfloat;
inline constexpr auto Float = vk::Format::eR32Sfloat;
inline constexpr auto Float2 = vk::Format::eR32G32Sfloat;
inline constexpr auto Uint = vk::Format::eR32Uint;
} // namespace Format

struct MainPipeline {
    MainPipeline(
        vk::Device, vk::SampleCountFlagBits,
        vk::DescriptorSetLayout shared_layout = {}, vk::DescriptorSet shared_set = {}
    );

    struct ResourcesT {
        ResourcesT(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::SampleCountFlagBits msaa_samples, vk::RenderPass);

        // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
        mvk::ImageResource DepthImage, OffscreenImage, ResolveImage;
        vk::UniqueFramebuffer Framebuffer;
    };

    void SetExtent(vk::Extent2D, vk::Device, vk::PhysicalDevice, vk::SampleCountFlagBits msaa_samples);

    PipelineRenderer Renderer;
    std::unique_ptr<ResourcesT> Resources;
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

struct ScenePipelines {
    ScenePipelines(vk::Device, vk::PhysicalDevice, vk::DescriptorSetLayout selection_layout = {}, vk::DescriptorSet selection_set = {});

    vk::Device Device;
    vk::PhysicalDevice PhysicalDevice;
    vk::SampleCountFlagBits Samples;

    MainPipeline Main;
    SilhouettePipeline Silhouette;
    SilhouetteEdgePipeline SilhouetteEdge;
    SelectionFragmentPipeline SelectionFragment;
    ComputePipeline ObjectPick, ElementPick, BoxSelect, UpdateSelectionState;
    IblPrefilterPipelines IblPrefilter;

    void SetExtent(vk::Extent2D);
    void CompileShaders();
};
