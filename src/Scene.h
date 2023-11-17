#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Camera.h"
#include "Geometry/GeometryMode.h"
#include "RenderMode.h"
#include "Shader.h"
#include "VulkanContext.h"

struct GeometryInstance;
struct GeometryBuffers;

struct ImageResource {
    // The `image` in the view info is overwritten.
    void Create(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Dereference forwards to the image.
    const vk::Image &operator*() const { return *Image; }

    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

enum class SelectionMode {
    None,
    Face,
    Vertex,
    Edge,
};

struct Transform {
    glm::mat4 Model;
    glm::mat4 View;
    glm::mat4 Projection;
};

struct ViewProjection {
    glm::mat4 View;
    glm::mat4 Projection;
};

struct ViewProjNearFar {
    glm::mat4 View;
    glm::mat4 Projection;
    float Near;
    float Far;
};

struct Light {
    glm::vec4 ColorAndAmbient; // RGB = color, A = ambient intensity. This is done for 16-byte alignment.
    glm::vec3 Direction;
};

struct SilhouetteControls {
    glm::vec4 Color;
    float Thickness;
    float Threshold;
};

struct Gizmo;

enum class ShaderPipelineType {
    Fill,
    Line,
    Grid,
    Silhouette,
    EdgeDetection,
    Combine,
};
using SPT = ShaderPipelineType;

struct RenderPipeline {
    RenderPipeline(const VulkanContext &);
    virtual ~RenderPipeline();

    // Updates images and framebuffer based on the new extent.
    // These resources are reused by future renders that don't change the extent.
    virtual void SetExtent(vk::Extent2D) = 0;

    void CompileShaders();

    void RenderGeometryBuffers(vk::CommandBuffer, const GeometryInstance &, SPT, GeometryMode) const;

    const VulkanContext &VC;

protected:
    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, std::unique_ptr<ShaderPipeline>> ShaderPipelines;
};

struct MainRenderPipeline : RenderPipeline {
    MainRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer, const vk::ClearColorValue &background_color) const;

    // All of the UBOs used in the pipeline.
    void UpdateDescriptors(
        vk::DescriptorBufferInfo transform,
        vk::DescriptorBufferInfo light,
        vk::DescriptorBufferInfo view_proj,
        vk::DescriptorBufferInfo view_proj_near_far
    ) const;

    void RenderGrid(vk::CommandBuffer) const;

    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;

    // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
    ImageResource DepthImage, OffscreenImage, ResolveImage;
};

struct SilhouetteRenderPipeline : RenderPipeline {
    SilhouetteRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;

    void UpdateDescriptors(vk::DescriptorBufferInfo transform) const;

    vk::Extent2D Extent;

    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
};

struct EdgeDetectionRenderPipeline : RenderPipeline {
    EdgeDetectionRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
    void Render(vk::CommandBuffer command_buffer) const;

    void UpdateDescriptors(vk::DescriptorBufferInfo silhouette_controls) const;
    void UpdateImageDescriptors(vk::DescriptorImageInfo silhouette_fill_image) const;

    vk::Extent2D Extent;

    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
};

struct FinalRenderPipeline : RenderPipeline {
    FinalRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
    void Render(vk::CommandBuffer command_buffer) const;

    void UpdateImageDescriptors(vk::DescriptorImageInfo main_scene_image, vk::DescriptorImageInfo silhouette_edge_image) const;

    vk::Extent2D Extent;

    // A single-sampled image resulting from combining the main & silhouette images.
    ImageResource OffscreenImage;
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MainRenderPipeline.MsaaSamples; }
    vk::ImageView GetResolveImageView() const { return *FinalRenderPipeline.OffscreenImage.View; }

    // Renders to a texture sampler and image view that can be accessed with `GetTextureSampler()` and `GetResolveImageView()`.
    // The extent of the resolve image can be found with `GetExtent()` after the call,
    // and it will be equal to the dimensions of `GetContentRegionAvail()` at the beginning of the call.
    // Returns true if the scene was updated, which can happen when the window size or background color changes.
    bool Render();
    void RenderGizmo();
    void RenderControls();

    // These do _not_ re-submit the command buffer. Callers must do so manually if needed.
    void CompileShaders();
    void UpdateTransform(); // Updates buffers that depend on model/view/transform.
    void UpdateGeometryEdgeColors();

    const VulkanContext &VC;

private:
    void SetExtent(vk::Extent2D);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;

    Camera Camera{{0, 0, 2}, Origin, 60, 0.1, 100};
    Light Light{{1, 1, 1, 0.6}, {0, 0, -1}}; // White light coming from the Z direction.

    glm::vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    glm::vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode Mode{RenderMode::FacesAndEdges};
    SelectionMode SelectionMode{SelectionMode::None};

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;

    VulkanBuffer TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)};
    VulkanBuffer LightBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Light)};
    VulkanBuffer ViewProjectionBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjection)};
    VulkanBuffer ViewProjNearFarBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)};
    VulkanBuffer SilhouetteControlsBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(SilhouetteControls)};

    vk::UniqueSampler MainSceneImageSampler, SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    MainRenderPipeline MainRenderPipeline;
    SilhouetteRenderPipeline SilhouetteRenderPipeline;
    FinalRenderPipeline FinalRenderPipeline;
    EdgeDetectionRenderPipeline EdgeDetectionRenderPipeline;
    std::vector<std::unique_ptr<RenderPipeline>> RenderPipelines;

    std::unique_ptr<Gizmo> Gizmo;
    std::vector<std::unique_ptr<GeometryInstance>> GeometryInstances;

    bool ShowGrid{true};
    SilhouetteControls SilhouetteControls{{1, 0.627, 0.157, 1.}, 2.f, 0.f}; // Color taken from Blender's default `Preferences->3D Viewport->Active Object`.
};
