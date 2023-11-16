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

struct Gizmo;

enum class ShaderPipelineType {
    Fill,
    Line,
    Grid,
    Silhouette,
};
using SPT = ShaderPipelineType;

struct RenderPipeline {
    RenderPipeline(const VulkanContext &);
    virtual ~RenderPipeline();

    void CompileShaders();

    const VulkanContext &VC;

protected:
    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, std::unique_ptr<ShaderPipeline>> ShaderPipelines;
};

struct MainRenderPipeline : RenderPipeline {
    MainRenderPipeline(const VulkanContext &);

    // All of the UBOs used in the pipeline.
    void UpdateDescriptors(
        vk::DescriptorBufferInfo transform,
        vk::DescriptorBufferInfo light,
        vk::DescriptorBufferInfo view_proj,
        vk::DescriptorBufferInfo view_proj_near_far
    ) const;

    // Recreates transform, render images (see below) and framebuffer based on the new extent.
    // These are then reused by future renders that don't change the extent.
    void SetExtent(vk::Extent2D);

    void Begin(const vk::UniqueCommandBuffer &, const vk::ClearColorValue &background_color) const;
    void RenderGrid(const vk::UniqueCommandBuffer &) const;
    void RenderGeometryBuffers(SPT, const vk::UniqueCommandBuffer &, const GeometryInstance &, GeometryMode) const;

    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;

    // We use three images in the render pass:
    // 1) Perform depth testing.
    // 2) Render into a multisampled offscreen image.
    // 3) Resolve into a single-sampled resolve image.
    // All images are referenced by the framebuffer and thus must be kept in memory.
    ImageResource DepthImage, OffscreenImage, ResolveImage;
};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MainRenderPipeline.MsaaSamples; }
    vk::ImageView GetResolveImageView() const { return MainRenderPipeline.ResolveImage.View.get(); }

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

    MainRenderPipeline MainRenderPipeline;

    std::unique_ptr<Gizmo> Gizmo;
    std::vector<std::unique_ptr<GeometryInstance>> GeometryInstances;

    bool ShowGrid{true};
};
