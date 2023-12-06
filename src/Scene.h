#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "Camera.h"
#include "Geometry/GeometryMode.h"
#include "RenderMode.h"
#include "Shader.h"
#include "VulkanContext.h"
#include "World.h"

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
    glm::mat4 Model{1};
    glm::mat4 View{1};
    glm::mat4 Projection{1};
    glm::mat4 NormalToWorld{1}; // Only a mat3 is needed, but mat4 is used for alignment.
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

struct Lights {
    // RGB: Color of the light emitting from the view position.
    // A: Ambient intensity. (Using a single vec4 for 16-byte alignment.)
    glm::vec4 ViewColorAndAmbient;
    // RGB: Color of the directional light.
    // A: Intensity of the directional light.
    glm::vec4 DirectionalColorAndIntensity;
    glm::vec3 Direction;
};

struct SilhouetteDisplay {
    glm::vec4 Color;
};

struct Gizmo;

enum class ShaderPipelineType {
    Fill,
    Line,
    Grid,
    Silhouette,
    EdgeDetection,
    Texture,
    DebugNormals,
};
using SPT = ShaderPipelineType;

struct RenderPipeline {
    struct ShaderBindingDescriptor {
        SPT PipelineType;
        std::string BindingName;
        std::optional<vk::DescriptorBufferInfo> BufferInfo{};
        std::optional<vk::DescriptorImageInfo> ImageInfo{};
    };

    RenderPipeline(const VulkanContext &);
    virtual ~RenderPipeline();

    // Updates images and framebuffer based on the new extent.
    // These resources are reused by future renders that don't change the extent.
    virtual void SetExtent(vk::Extent2D) = 0;

    inline const ShaderPipeline *GetShaderPipeline(SPT spt) const { return ShaderPipelines.at(spt).get(); }
    void CompileShaders();

    void UpdateDescriptors(std::vector<ShaderBindingDescriptor> &&) const;

    void RenderBuffers(vk::CommandBuffer, const GeometryBuffers &, SPT) const;

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

    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;

    // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
    ImageResource DepthImage, OffscreenImage, ResolveImage;
};

struct SilhouetteRenderPipeline : RenderPipeline {
    SilhouetteRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;

    vk::Extent2D Extent;

    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
};

struct EdgeDetectionRenderPipeline : RenderPipeline {
    EdgeDetectionRenderPipeline(const VulkanContext &);

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;

    vk::Extent2D Extent;

    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.
};

inline static const Camera DefaultCamera{{0, 0, 2}, World::Origin, 60, 0.1, 100};

struct Scene {
    Scene(const VulkanContext &);
    ~Scene();

    const vk::Extent2D &GetExtent() const { return Extent; }
    vk::SampleCountFlagBits GetMsaaSamples() const { return MainRenderPipeline.MsaaSamples; }
    vk::ImageView GetResolveImageView() const { return *MainRenderPipeline.ResolveImage.View; }

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
    void UpdateEdgeColors();
    void UpdateNormalIndicators();

    const VulkanContext &VC;

private:
    void SetExtent(vk::Extent2D);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;
    void RecordAndSubmitCommandBuffer(vk::Fence fence = nullptr);

    Camera Camera{DefaultCamera};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};

    glm::vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    glm::vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};
    SelectionMode SelectionMode{SelectionMode::None};
    bool ShowFaceNormals{false}, ShowVertexNormals{false};

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;

    VulkanBuffer TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)};
    VulkanBuffer LightsBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Lights)};
    VulkanBuffer ViewProjectionBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjection)};
    VulkanBuffer ViewProjNearFarBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)};
    VulkanBuffer SilhouetteDisplayBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(SilhouetteDisplay)};

    vk::UniqueSampler SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    MainRenderPipeline MainRenderPipeline;
    SilhouetteRenderPipeline SilhouetteRenderPipeline;
    EdgeDetectionRenderPipeline EdgeDetectionRenderPipeline;
    std::vector<std::unique_ptr<RenderPipeline>> RenderPipelines;

    std::unique_ptr<Gizmo> Gizmo;
    std::vector<std::unique_ptr<GeometryInstance>> GeometryInstances;

    bool ShowGrid{true};
    SilhouetteDisplay SilhouetteDisplay{{1, 0.627, 0.157, 1.}}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
    glm::vec4 BgColor{0.22, 0.22, 0.22, 1.};
};
