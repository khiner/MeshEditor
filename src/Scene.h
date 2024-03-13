#pragma once

#include <unordered_set>

#include <entt/entity/registry.hpp>

#include "numeric/mat4.h"
#include "numeric/vec2.h"
#include "numeric/vec3.h"
#include "numeric/vec4.h"

#include "Camera.h"
#include "RenderMode.h"
#include "Shader.h"
#include "World.h"
#include "mesh/MeshElement.h"
#include "vulkan/VulkanBuffer.h"

struct Mesh;
struct VulkanContext;
struct MeshBuffers;
struct MeshElementBuffers;
struct VkBuffers;

struct ImageResource {
    // The `image` in the view info is overwritten.
    void Create(const VulkanContext &, vk::ImageCreateInfo, vk::ImageViewCreateInfo, vk::MemoryPropertyFlags flags = vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Dereference forwards to the image.
    const vk::Image &operator*() const { return *Image; }

    vk::UniqueImage Image;
    vk::UniqueImageView View;
    vk::UniqueDeviceMemory Memory;
};

struct Transform {
    mat4 View{1};
    mat4 Projection{1};
    mat4 NormalToWorld{1}; // Only a mat3 is needed, but mat4 is used for alignment.
};

struct ViewProjNearFar {
    mat4 View, Projection;
    float Near, Far;
};

struct Lights {
    // RGB: Color of the light emitting from the view position.
    // A: Ambient intensity. (Using a single vec4 for 16-byte alignment.)
    vec4 ViewColorAndAmbient;
    // RGB: Color of the directional light.
    // A: Intensity of the directional light.
    vec4 DirectionalColorAndIntensity;
    vec3 Direction;
};

struct SilhouetteDisplay {
    vec4 Color;
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

    const VulkanContext &VC;

    // Updates images and framebuffer based on the new extent.
    // These resources are reused by future renders that don't change the extent.
    virtual void SetExtent(vk::Extent2D) = 0;

    const ShaderPipeline *GetShaderPipeline(SPT spt) const { return ShaderPipelines.at(spt).get(); }
    void CompileShaders();

    void UpdateDescriptors(std::vector<ShaderBindingDescriptor> &&) const;

    void RenderBuffers(vk::CommandBuffer, const MeshBuffers &, SPT, const VulkanBuffer &models_buffer) const;

protected:
    vk::UniqueFramebuffer Framebuffer;
    vk::UniqueRenderPass RenderPass;
    std::unordered_map<SPT, std::unique_ptr<ShaderPipeline>> ShaderPipelines;
};

struct MainRenderPipeline : RenderPipeline {
    MainRenderPipeline(const VulkanContext &);

    vk::SampleCountFlagBits MsaaSamples;
    vk::Extent2D Extent;
    // Perform depth testing, render into a multisampled offscreen image, and resolve into a single-sampled image.
    ImageResource DepthImage, OffscreenImage, ResolveImage;

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer, const vk::ClearColorValue &background_color) const;
};

struct SilhouetteRenderPipeline : RenderPipeline {
    SilhouetteRenderPipeline(const VulkanContext &);

    vk::Extent2D Extent;
    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

struct EdgeDetectionRenderPipeline : RenderPipeline {
    EdgeDetectionRenderPipeline(const VulkanContext &);

    vk::Extent2D Extent;
    ImageResource OffscreenImage; // Single-sampled image without a depth buffer.

    void SetExtent(vk::Extent2D) override;
    void Begin(vk::CommandBuffer) const;
};

struct Scene {
    Scene(const VulkanContext &, entt::registry &);
    ~Scene();

    const VulkanContext &VC;
    entt::registry &R;

    void AddMesh(Mesh &&);

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

    Camera CreateDefaultCamera() const { return {World.Up, {0, 0, 2}, World.Origin, 60, 0.1, 100}; }

private:
    World World{};
    Camera Camera{CreateDefaultCamera()};
    Lights Lights{{1, 1, 1, 0.1}, {1, 1, 1, 0.15}, {-1, -1, -1}};
    MeshElementIndex HighlightedElement{};

    vec4 EdgeColor{1, 1, 1, 1}; // Used for line mode.
    vec4 MeshEdgeColor{0, 0, 0, 1}; // Used for faces mode.

    RenderMode RenderMode{RenderMode::FacesAndEdges};
    ColorMode ColorMode{ColorMode::Mesh};
    MeshElement SelectionElement{MeshElement::None};
    std::unordered_set<MeshElement> ShownNormals{};

    entt::entity SelectedEntity{0};
    // Map of entities to contiguous indices.
    // Currently only used for `MeshElementBuffers` indexing, but could be used for anything.
    std::unordered_map<entt::entity, uint> MeshBufferIndices;
    std::unique_ptr<MeshElementBuffers> MeshElementBuffers;

    vk::Extent2D Extent;
    vk::ClearColorValue BackgroundColor;

    VulkanBuffer
        TransformBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Transform)},
        LightsBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(Lights)},
        ViewProjNearFarBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(ViewProjNearFar)},
        SilhouetteDisplayBuffer{vk::BufferUsageFlagBits::eUniformBuffer, sizeof(SilhouetteDisplay)},
        ModelsBuffer{vk::BufferUsageFlagBits::eVertexBuffer};

    vk::UniqueSampler SilhouetteFillImageSampler, SilhouetteEdgeImageSampler;

    MainRenderPipeline MainRenderPipeline;
    SilhouetteRenderPipeline SilhouetteRenderPipeline;
    EdgeDetectionRenderPipeline EdgeDetectionRenderPipeline;
    std::vector<std::unique_ptr<RenderPipeline>> RenderPipelines;

    std::unique_ptr<Gizmo> Gizmo;

    bool ShowGrid{true};
    SilhouetteDisplay SilhouetteDisplay{{1, 0.627, 0.157, 1.}}; // Blender's default `Preferences->Themes->3D Viewport->Active Object`.
    vec4 BgColor{0.22, 0.22, 0.22, 1.};

    Mesh &GetSelectedMesh() const;
    mat4 &GetSelectedModel() const;

    void SetExtent(vk::Extent2D);
    void RecordCommandBuffer();
    void SubmitCommandBuffer(vk::Fence fence = nullptr) const;
    void RecordAndSubmitCommandBuffer(vk::Fence fence = nullptr);

    void CreateOrUpdateBuffers(entt::entity, MeshElementIndex highlight_element = {});
};
